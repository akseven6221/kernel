"""Compare the CuTe port of CUDA v6 with the original PTX and cuBLAS on B200."""
from pathlib import Path

import modal

SOURCE_DIR = Path(__file__).resolve().parent
CUDA_DIR = SOURCE_DIR.parent / "cuda_kernel"
CUDA_IMAGE = "nvidia/cuda:12.8.1-devel-ubuntu24.04"
CUTLASS_COMMIT = "f3fde58372d33e9a5650ba7b80fc48b3b49d40c8"  # v4.2.1
app = modal.App("write-myself2-v5-double-buffer-b200")
image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.12")
    .apt_install("curl")
    .run_commands(
        f"curl --fail --location --retry 3 https://codeload.github.com/NVIDIA/cutlass/tar.gz/{CUTLASS_COMMIT} -o /tmp/cutlass.tar.gz",
        "mkdir -p /opt/cutlass",
        "tar -xzf /tmp/cutlass.tar.gz -C /opt/cutlass --strip-components=1",
        "rm /tmp/cutlass.tar.gz",
    )
    .add_local_file(SOURCE_DIR / "v5.cu", "/root/user_v5.cu")
    .add_local_file(CUDA_DIR / "v6.cu", "/root/v6.cu")
    .add_local_file(CUDA_DIR / "benchmark_v6.cu", "/root/benchmark_original.cu")
)


@app.function(image=image, gpu="B200", timeout=1800)
def benchmark(m: int, n: int, k: int, sanitize: bool, mma_n: int) -> dict:
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
    Path("/root/write_v5.cu").write_text(
        '#include "user_v5.cu"\nnamespace mojo_v6 = v5;\n'
    )
    original = Path("/root/benchmark_original.cu").read_text()
    variants = ("v6", "write_v5")
    for variant in variants:
        source = original.replace('#include "v6.cu"', f'#include "{variant}.cu"', 1)
        source = source.replace("CUDA v6:", f"{variant}:")
        Path(f"/root/benchmark_{variant}.cu").write_text(source)
    run(["nvidia-smi"])
    run(["nvcc", "--version"])
    for variant in variants:
        common = ["nvcc", "-std=c++17", "-O3", "-lineinfo", "-arch=sm_100a",
                  "-I/opt/cutlass/include", f"-DMOJO_V6_MMA_N={mma_n}",
                  f"/root/benchmark_{variant}.cu"]
        run(common + ["-Xptxas=-v", "-o", f"/root/benchmark_{variant}",
                      "-L/usr/local/cuda/lib64/stubs", "-lcuda", "-lcublas"])
        run(common + ["--ptx", "-o", f"/root/{variant}.ptx"])

    # Match each configuration's ring depth and test both sides of each wrap.
    stages = 6 if mma_n == 256 else 8
    correctness_shapes = [(256, mma_n, 64), (512, 3 * mma_n, 128),
                          (768, 2 * mma_n, 192)]
    correctness_shapes += [(512, 3 * mma_n, iterations * 64) for iterations in
                           (stages - 1, stages, stages + 1, 2 * stages - 1,
                            2 * stages, 2 * stages + 1, 3 * stages + 1)]
    correctness_shapes.append((4096, 4096, 64))
    for shape in correctness_shapes:
        for variant in variants:
            run([f"/root/benchmark_{variant}", "--correctness-only", *map(str, shape)])
    if sanitize:
        sanitizer_shape = (768, 3 * mma_n, (2 * stages + 1) * 64)
        large_sanitizer_shape = (4096, 4096, (3 * stages + 1) * 64)
        run(["compute-sanitizer", "--version"])
        for tool in ("memcheck", "synccheck", "racecheck"):
            run(["compute-sanitizer", "--tool", tool, "--error-exitcode", "1",
                 "/root/benchmark_write_v5", "--correctness-only", *map(str, sanitizer_shape)],
                timeout=300)
        run(["compute-sanitizer", "--tool", "racecheck", "--error-exitcode", "1",
             "/root/benchmark_write_v5", "--correctness-only", *map(str, large_sanitizer_shape)],
            timeout=300)

    runs = {variant: [] for variant in variants}
    for order in (("v6", "write_v5"), ("write_v5", "v6"), ("v6", "write_v5")):
        for variant in order:
            output = run([f"/root/benchmark_{variant}", str(m), str(n), str(k)], timeout=300)
            runs[variant].append(next(json.loads(line.removeprefix("RESULT_JSON="))
                                      for line in output.splitlines()
                                      if line.startswith("RESULT_JSON=")))
    medians = {variant: statistics.median(r["custom_ms"] for r in values)
               for variant, values in runs.items()}
    custom_ms = medians["write_v5"]
    cublas_ms = statistics.median(r["cublas_ms"] for r in runs["write_v5"])
    operations = 2.0 * m * n * k
    return {
        "m": m, "n": n, "k": k, "mma_n": mma_n, "ab_stages": stages,
        "ptx_ms": medians["v6"], "custom_ms": custom_ms, "cublas_ms": cublas_ms,
        "custom_tflops": operations / custom_ms / 1e9,
        "ptx_tflops": operations / medians["v6"] / 1e9,
        "cublas_tflops": operations / cublas_ms / 1e9,
        "throughput_percent": 100 * cublas_ms / custom_ms,
        "latency_change_vs_ptx_percent": (custom_ms / medians["v6"] - 1) * 100,
        "runs": runs, "sanitize": sanitize, "correctness_shapes": correctness_shapes,
        "sanitizer_scope": "unfiltered: custom kernels and cuBLAS" if sanitize else "not run",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "cuda_image": CUDA_IMAGE, "cutlass_version": "4.2.1", "cutlass_commit": CUTLASS_COMMIT,
        "timing": "AB/BA/AB; each run: 100 direct warmups, 50 alternating warmup graph replays each, 100 GEMMs/replay, CUDA events, alternating custom/cuBLAS, median of 7 rounds; final median of 3 run medians",
        "baseline": "unchanged cuda_kernel/v6.cu; cublasGemmEx BF16 A/B/C, FP32 accumulation, reduced-precision reduction disabled",
        "source_sha256": {name: hashlib.sha256(Path("/root", name).read_bytes()).hexdigest()
                          for name in ("v6.cu", "user_v5.cu", "write_v5.cu", "benchmark_original.cu",
                                       "benchmark_v6.cu", "benchmark_write_v5.cu")},
        "generated_ptx": {v: Path(f"/root/{v}.ptx").read_text() for v in variants},
        "logs": logs,
    }


@app.local_entrypoint()
def main(m: int = 4096, n: int = 4096, k: int = 4096,
         sanitize: bool = False, mma_n: int = 256):
    import hashlib
    import json

    if mma_n not in (128, 256):
        raise ValueError("mma_n must be 128 or 256")
    if min(m, n, k) <= 0 or m % 256 or n % mma_n or k % 64:
        raise ValueError(f"Required: positive M,N,K; M % 256 = N % {mma_n} = K % 64 = 0")
    result = benchmark.remote(m, n, k, sanitize, mma_n)
    for local_path, remote_name in ((SOURCE_DIR / "v5.cu", "user_v5.cu"),
                                    (CUDA_DIR / "v6.cu", "v6.cu"),
                                    (CUDA_DIR / "benchmark_v6.cu", "benchmark_original.cu")):
        if hashlib.sha256(local_path.read_bytes()).hexdigest() != result["source_sha256"][remote_name]:
            raise RuntimeError(f"Source changed during benchmark: {local_path}")
    suffix = "" if mma_n == 256 else "_mma128"
    for variant, ptx in result.pop("generated_ptx").items():
        name = "ptx_v6" if variant == "v6" else variant
        (SOURCE_DIR / f"{name}{suffix}.sm100a.ptx").write_text(ptx)
    result_path = SOURCE_DIR / f"results_v5_b200_{m}x{n}x{k}{suffix}.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {result_path}")
    print(f"CuTe {result['custom_ms']:.6f} ms ({result['custom_tflops']:.2f} TFLOPS); "
          f"PTX {result['ptx_ms']:.6f} ms; cuBLAS {result['cublas_ms']:.6f} ms; "
          f"CuTe/cuBLAS throughput {result['throughput_percent']:.2f}%; "
          f"latency vs PTX {result['latency_change_vs_ptx_percent']:+.2f}%")
