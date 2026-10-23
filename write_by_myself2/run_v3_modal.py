"""Compare the CuTe port of CUDA v4 with the original PTX and cuBLAS on B200."""
from pathlib import Path

import modal

SOURCE_DIR = Path(__file__).resolve().parent
CUDA_DIR = SOURCE_DIR.parent / "cuda_kernel"
CUDA_IMAGE = "nvidia/cuda:12.8.1-devel-ubuntu24.04"
CUTLASS_COMMIT = "f3fde58372d33e9a5650ba7b80fc48b3b49d40c8"  # v4.2.1
app = modal.App("write-myself2-v3-2sm-b200")
image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.12")
    .apt_install("curl")
    .run_commands(
        f"curl --fail --location --retry 3 https://codeload.github.com/NVIDIA/cutlass/tar.gz/{CUTLASS_COMMIT} -o /tmp/cutlass.tar.gz",
        "mkdir -p /opt/cutlass",
        "tar -xzf /tmp/cutlass.tar.gz -C /opt/cutlass --strip-components=1",
        "rm /tmp/cutlass.tar.gz",
    )
    .add_local_file(SOURCE_DIR / "v3.cu", "/root/user_v3.cu")
    .add_local_file(CUDA_DIR / "v4.cu", "/root/v4.cu")
    .add_local_file(CUDA_DIR / "benchmark_v4.cu", "/root/benchmark_original.cu")
)


@app.function(image=image, gpu="B200", timeout=1800)
def benchmark(m: int, n: int, k: int, sanitize: bool) -> dict:
    import datetime
    import hashlib
    import json
    import os
    import signal
    import statistics
    import subprocess

    logs = []

    def run(command: list[str], timeout: int = 180) -> str:
        print("$ " + " ".join(command), flush=True)
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, start_new_session=True,
        )
        try:
            output, _ = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate()
            print(output, flush=True)
            raise
        print(output, flush=True)
        logs.append({"command": command, "output": output})
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command, output)
        return output

    # Preserve the original validation, cuBLAS settings and timing procedure.
    Path("/root/write_v3.cu").write_text(
        '#include "user_v3.cu"\nnamespace mojo_v4 = v3;\n'
    )
    original = Path("/root/benchmark_original.cu").read_text()
    variants = ("v4", "write_v3")
    for variant in variants:
        source = original.replace('#include "v4.cu"', f'#include "{variant}.cu"', 1)
        source = source.replace("CUDA v4:", f"{variant}:")
        Path(f"/root/benchmark_{variant}.cu").write_text(source)
    run(["nvidia-smi"])
    run(["nvcc", "--version"])
    for variant in variants:
        common = ["nvcc", "-std=c++17", "-O3", "-lineinfo", "-arch=sm_100a",
                  "-I/opt/cutlass/include", f"/root/benchmark_{variant}.cu"]
        run(common + ["-Xptxas=-v", "-o", f"/root/benchmark_{variant}",
                      "-L/usr/local/cuda/lib64/stubs", "-lcuda", "-lcublas"])
        run(common + ["--ptx", "-o", f"/root/{variant}.ptx"])

    correctness_shapes = [(256, 256, 64), (512, 512, 192), (768, 768, 576),
                          (768, 512, 576), (512, 768, 128), (4096, 4096, 64)]
    for shape in correctness_shapes:
        for variant in variants:
            run([f"/root/benchmark_{variant}", "--correctness-only", *map(str, shape)])
    if sanitize:
        run(["compute-sanitizer", "--version"])
        for tool in ("memcheck", "synccheck", "racecheck"):
            run(["compute-sanitizer", "--tool", tool, "--error-exitcode", "1",
                 "/root/benchmark_write_v3", "--correctness-only", "768", "512", "576"],
                timeout=300)
        run(["compute-sanitizer", "--tool", "racecheck", "--error-exitcode", "1",
             "/root/benchmark_write_v3", "--correctness-only", "4096", "4096", "576"],
            timeout=300)

    runs = {variant: [] for variant in variants}
    for order in (("v4", "write_v3"), ("write_v3", "v4"), ("v4", "write_v3")):
        for variant in order:
            output = run([f"/root/benchmark_{variant}", str(m), str(n), str(k)], timeout=300)
            runs[variant].append(next(json.loads(line.removeprefix("RESULT_JSON="))
                                      for line in output.splitlines()
                                      if line.startswith("RESULT_JSON=")))
    medians = {variant: statistics.median(r["custom_ms"] for r in values)
               for variant, values in runs.items()}
    custom_ms = medians["write_v3"]
    cublas_ms = statistics.median(r["cublas_ms"] for r in runs["write_v3"])
    operations = 2.0 * m * n * k
    return {
        "m": m, "n": n, "k": k,
        "ptx_ms": medians["v4"], "custom_ms": custom_ms, "cublas_ms": cublas_ms,
        "custom_tflops": operations / custom_ms / 1e9,
        "ptx_tflops": operations / medians["v4"] / 1e9,
        "cublas_tflops": operations / cublas_ms / 1e9,
        "throughput_percent": 100 * cublas_ms / custom_ms,
        "latency_change_vs_ptx_percent": (custom_ms / medians["v4"] - 1) * 100,
        "runs": runs, "sanitize": sanitize, "correctness_shapes": correctness_shapes,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "cuda_image": CUDA_IMAGE, "cutlass_version": "4.2.1", "cutlass_commit": CUTLASS_COMMIT,
        "timing": "AB/BA/AB; each run: 100 direct warmups, 50 alternating warmup graph replays each, 100 GEMMs/replay, CUDA events, alternating custom/cuBLAS, median of 7 rounds; final median of 3 run medians",
        "baseline": "unchanged cuda_kernel/v4.cu; cublasGemmEx BF16 A/B/C, FP32 accumulation, reduced-precision reduction disabled",
        "source_sha256": {name: hashlib.sha256(Path("/root", name).read_bytes()).hexdigest()
                          for name in ("v4.cu", "user_v3.cu", "write_v3.cu", "benchmark_original.cu",
                                       "benchmark_v4.cu", "benchmark_write_v3.cu")},
        "generated_ptx": {v: Path(f"/root/{v}.ptx").read_text() for v in variants},
        "logs": logs,
    }


@app.local_entrypoint()
def main(m: int = 4096, n: int = 4096, k: int = 4096, sanitize: bool = False):
    import hashlib
    import json

    if min(m, n, k) <= 0 or m % 256 or n % 256 or k % 64:
        raise ValueError("Required: positive M,N,K; M % 256 = N % 256 = K % 64 = 0")
    result = benchmark.remote(m, n, k, sanitize)
    for local_path, remote_name in ((SOURCE_DIR / "v3.cu", "user_v3.cu"),
                                    (CUDA_DIR / "v4.cu", "v4.cu"),
                                    (CUDA_DIR / "benchmark_v4.cu", "benchmark_original.cu")):
        if hashlib.sha256(local_path.read_bytes()).hexdigest() != result["source_sha256"][remote_name]:
            raise RuntimeError(f"Source changed during benchmark: {local_path}")
    for variant, ptx in result.pop("generated_ptx").items():
        name = "ptx_v4" if variant == "v4" else variant
        (SOURCE_DIR / f"{name}.sm100a.ptx").write_text(ptx)
    result_path = SOURCE_DIR / f"results_v3_b200_{m}x{n}x{k}.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {result_path}")
    print(f"CuTe {result['custom_ms']:.6f} ms ({result['custom_tflops']:.2f} TFLOPS); "
          f"PTX {result['ptx_ms']:.6f} ms; cuBLAS {result['cublas_ms']:.6f} ms; "
          f"CuTe/cuBLAS throughput {result['throughput_percent']:.2f}%; "
          f"latency vs PTX {result['latency_change_vs_ptx_percent']:+.2f}%")
