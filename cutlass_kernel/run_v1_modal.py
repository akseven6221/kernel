"""Instruction-level CUTLASS port vs unchanged PTX v1_new on the same Modal B200."""
from pathlib import Path

import modal

SOURCE_DIR = Path(__file__).resolve().parent
CUDA_DIR = SOURCE_DIR.parent / "cuda_kernel"
CUDA_IMAGE = "nvidia/cuda:12.8.1-devel-ubuntu24.04"
CUTLASS_COMMIT = "f3fde58372d33e9a5650ba7b80fc48b3b49d40c8"  # v4.2.1
app = modal.App("cutlass-v1-vs-ptx-b200")
image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.12")
    .apt_install("curl")
    .run_commands(
        f"curl --fail --location --retry 3 https://codeload.github.com/NVIDIA/cutlass/tar.gz/{CUTLASS_COMMIT} -o /tmp/cutlass.tar.gz",
        "mkdir -p /opt/cutlass",
        "tar -xzf /tmp/cutlass.tar.gz -C /opt/cutlass --strip-components=1",
        "rm /tmp/cutlass.tar.gz",
    )
    .add_local_file(SOURCE_DIR / "v1.cu", "/root/cutlass_v1.cu")
    .add_local_file(CUDA_DIR / "v1_new.cu", "/root/v1_new.cu")
    .add_local_file(CUDA_DIR / "benchmark.cu", "/root/benchmark.cu")
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

    run(["nvidia-smi"])
    run(["nvcc", "--version"])
    for variant in ["v1_new", "cutlass_v1"]:
        common = ["nvcc", "-std=c++17", "-O3", "-lineinfo", "-arch=sm_100a",
                  "-I/opt/cutlass/include", f'-DGEMM_KERNEL="{variant}.cu"',
                  f'-DGEMM_LABEL="{variant}"', "/root/benchmark.cu"]
        run(common + ["-Xptxas=-v", "-o", f"/root/benchmark_{variant}",
                      "-L/usr/local/cuda/lib64/stubs", "-lcuda", "-lcublas"])
        run(common + ["--ptx", "-o", f"/root/{variant}.ptx"])
    correctness_shapes = [(64, 256, 64), (128, 512, 192), (192, 256, 512),
                          (256, 768, 1024)]
    for shape in correctness_shapes:
        for variant in ["v1_new", "cutlass_v1"]:
            run([f"/root/benchmark_{variant}", "--correctness-only", *map(str, shape)])
    if sanitize:
        for tool in ["memcheck", "synccheck", "racecheck"]:
            run(["compute-sanitizer", "--tool", tool, "--error-exitcode", "1",
                 "/root/benchmark_cutlass_v1", "--correctness-only", "128", "512", "192"],
                timeout=300)
    # AB / BA / AB balances slow environmental drift; every run also measures cuBLAS.
    runs = {"v1_new": [], "cutlass_v1": []}
    for order in [("v1_new", "cutlass_v1"), ("cutlass_v1", "v1_new"),
                  ("v1_new", "cutlass_v1")]:
        for variant in order:
            output = run([f"/root/benchmark_{variant}", str(m), str(n), str(k)], timeout=300)
            runs[variant].append(next(json.loads(line.removeprefix("RESULT_JSON="))
                                      for line in output.splitlines()
                                      if line.startswith("RESULT_JSON=")))
    medians = {variant: statistics.median(r["custom_ms"] for r in values)
               for variant, values in runs.items()}
    result = {
        "m": m, "n": n, "k": k,
        "ptx_ms": medians["v1_new"], "cutlass_ms": medians["cutlass_v1"],
        "latency_change_percent": (medians["cutlass_v1"] / medians["v1_new"] - 1) * 100,
        "speedup": medians["v1_new"] / medians["cutlass_v1"],
        "runs": runs, "sanitize": sanitize, "correctness_shapes": correctness_shapes,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "cuda_image": CUDA_IMAGE, "cutlass_version": "4.2.1", "cutlass_commit": CUTLASS_COMMIT,
        "timing": "AB/BA/AB; each run: CUDA Graph, 100 GEMMs/replay, CUDA events, alternating custom/cuBLAS, median of 7 rounds; final median of 3 run medians",
        "baseline": "unchanged cuda_kernel/v1_new.cu; cublasGemmEx FP32 accumulation, reduced-precision reduction disabled",
        "source_sha256": {name: hashlib.sha256(Path("/root", name).read_bytes()).hexdigest()
                          for name in ["v1_new.cu", "cutlass_v1.cu", "benchmark.cu"]},
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
    for variant, ptx in result.pop("generated_ptx").items():
        (SOURCE_DIR / f"{variant}.sm100a.ptx").write_text(ptx)
    result_path = SOURCE_DIR / f"results_v1_b200_{m}x{n}x{k}.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {result_path}")
    print(f"PTX {result['ptx_ms']:.6f} ms; CUTLASS {result['cutlass_ms']:.6f} ms; "
          f"latency change {result['latency_change_percent']:+.2f}%; speedup {result['speedup']:.4f}x")
