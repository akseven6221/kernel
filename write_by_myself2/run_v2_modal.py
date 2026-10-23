"""Validate the CuTe v2 TMA epilogue against cuda_kernel/v3.cu on Modal B200."""
from pathlib import Path

import modal

SOURCE_DIR = Path(__file__).resolve().parent
CUDA_DIR = SOURCE_DIR.parent / "cuda_kernel"
CUDA_IMAGE = "nvidia/cuda:12.8.1-devel-ubuntu24.04"
CUTLASS_COMMIT = "f3fde58372d33e9a5650ba7b80fc48b3b49d40c8"  # v4.2.1
app = modal.App("write-myself2-v2-tma-b200")
image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.12")
    .apt_install("curl")
    .run_commands(
        f"curl --fail --location --retry 3 https://codeload.github.com/NVIDIA/cutlass/tar.gz/{CUTLASS_COMMIT} -o /tmp/cutlass.tar.gz",
        "mkdir -p /opt/cutlass",
        "tar -xzf /tmp/cutlass.tar.gz -C /opt/cutlass --strip-components=1",
        "rm /tmp/cutlass.tar.gz",
    )
    .add_local_file(SOURCE_DIR / "v2.cu", "/root/user_v2.cu")
    .add_local_file(CUDA_DIR / "v3.cu", "/root/v3.cu")
    .add_local_file(CUDA_DIR / "benchmark_v3.cu", "/root/benchmark_original.cu")
)


@app.function(image=image, gpu="B200", timeout=1800)
def benchmark(m: int, n: int, k: int, sanitize: bool) -> dict:
    import datetime
    import hashlib
    import json
    import statistics
    import subprocess

    logs = []

    def run(command: list[str], timeout: int = 180) -> str:
        print("$ " + " ".join(command), flush=True)
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, timeout=timeout)
        print(process.stdout, flush=True)
        logs.append({"command": command, "output": process.stdout})
        process.check_returncode()
        return process.stdout

    # Adapt the existing v3 benchmark without changing its validation or timing.
    Path("/root/write_v2.cu").write_text('#include "user_v2.cu"\nnamespace mojo_v3 = v2;\n')
    original = Path("/root/benchmark_original.cu").read_text()
    for variant in ["v3", "write_v2"]:
        source = original.replace('#include "v3.cu"', f'#include "{variant}.cu"', 1)
        source = source.replace('CUDA v3:', f'{variant}:')
        Path(f"/root/benchmark_{variant}.cu").write_text(source)
    run(["nvidia-smi"])
    run(["nvcc", "--version"])
    for variant in ["v3", "write_v2"]:
        common = ["nvcc", "-std=c++17", "-O3", "-lineinfo", "-arch=sm_100a",
                  "-I/opt/cutlass/include", f"/root/benchmark_{variant}.cu"]
        run(common + ["-Xptxas=-v", "-o", f"/root/benchmark_{variant}",
                      "-L/usr/local/cuda/lib64/stubs", "-lcuda", "-lcublas"])
        run(common + ["--ptx", "-o", f"/root/{variant}.ptx"])
    correctness_shapes = [(64, 256, 64), (128, 512, 192), (192, 768, 576),
                          (4096, 4096, 64)]
    for shape in correctness_shapes:
        for variant in ["v3", "write_v2"]:
            run([f"/root/benchmark_{variant}", "--correctness-only", *map(str, shape)])
    if sanitize:
        for tool in ["memcheck", "synccheck", "racecheck"]:
            run(["compute-sanitizer", "--tool", tool, "--error-exitcode", "1",
                 "/root/benchmark_write_v2", "--correctness-only", "192", "512", "576"],
                timeout=300)
        run(["compute-sanitizer", "--tool", "racecheck", "--error-exitcode", "1",
             "/root/benchmark_write_v2", "--correctness-only", "4096", "4096", "576"],
            timeout=300)
    # AB / BA / AB balances slow environmental drift; every run also measures cuBLAS.
    runs = {"v3": [], "write_v2": []}
    for order in [("v3", "write_v2"), ("write_v2", "v3"),
                  ("v3", "write_v2")]:
        for variant in order:
            output = run([f"/root/benchmark_{variant}", str(m), str(n), str(k)], timeout=300)
            runs[variant].append(next(json.loads(line.removeprefix("RESULT_JSON="))
                                      for line in output.splitlines()
                                      if line.startswith("RESULT_JSON=")))
    medians = {variant: statistics.median(r["custom_ms"] for r in values)
               for variant, values in runs.items()}
    result = {
        "m": m, "n": n, "k": k,
        "ptx_ms": medians["v3"], "custom_ms": medians["write_v2"],
        "latency_change_percent": (medians["write_v2"] / medians["v3"] - 1) * 100,
        "speedup": medians["v3"] / medians["write_v2"],
        "runs": runs, "sanitize": sanitize, "correctness_shapes": correctness_shapes,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "cuda_image": CUDA_IMAGE, "cutlass_version": "4.2.1", "cutlass_commit": CUTLASS_COMMIT,
        "timing": "AB/BA/AB; each run: 100 direct warmups, 50 alternating warmup graph replays each, CUDA Graph, 100 GEMMs/replay, CUDA events, alternating custom/cuBLAS, median of 7 rounds; final median of 3 run medians",
        "baseline": "unchanged cuda_kernel/v3.cu; cublasGemmEx FP32 accumulation, reduced-precision reduction disabled",
        "source_sha256": {name: hashlib.sha256(Path("/root", name).read_bytes()).hexdigest()
                          for name in ["v3.cu", "user_v2.cu", "write_v2.cu", "benchmark_original.cu",
                                       "benchmark_v3.cu", "benchmark_write_v2.cu"]},
        "generated_ptx": {v: Path(f"/root/{v}.ptx").read_text() for v in runs},
        "logs": logs,
    }
    return result


@app.local_entrypoint()
def main(m: int = 4096, n: int = 4096, k: int = 4096, sanitize: bool = False):
    import json

    if min(m, n, k) <= 0 or m % 64 or n % 256 or k % 64:
        raise ValueError("Required: positive M,N,K; M % 64 = N % 256 = K % 64 = 0")
    result = benchmark.remote(m, n, k, sanitize)
    import hashlib

    for local_path, remote_name in [(SOURCE_DIR / "v2.cu", "user_v2.cu"),
                                    (CUDA_DIR / "v3.cu", "v3.cu"),
                                    (CUDA_DIR / "benchmark_v3.cu", "benchmark_original.cu")]:
        if hashlib.sha256(local_path.read_bytes()).hexdigest() != result["source_sha256"][remote_name]:
            raise RuntimeError(f"Source changed during benchmark: {local_path}")
    for variant, ptx in result.pop("generated_ptx").items():
        (SOURCE_DIR / f"{variant}.sm100a.ptx").write_text(ptx)
    result_path = SOURCE_DIR / f"results_v2_b200_{m}x{n}x{k}.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {result_path}")
    print(f"PTX {result['ptx_ms']:.6f} ms; write_by_myself2/v2 {result['custom_ms']:.6f} ms; "
          f"latency change {result['latency_change_percent']:+.2f}%; speedup {result['speedup']:.4f}x")


