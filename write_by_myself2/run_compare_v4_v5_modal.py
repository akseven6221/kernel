"""Compare CuTe v4/v5 on one B200 using balanced, alternating benchmark runs."""
from pathlib import Path

import modal

SOURCE_DIR = Path(__file__).resolve().parent
CUDA_DIR = SOURCE_DIR.parent / "cuda_kernel"
CUDA_IMAGE = "nvidia/cuda:12.8.1-devel-ubuntu24.04"
CUTLASS_COMMIT = "f3fde58372d33e9a5650ba7b80fc48b3b49d40c8"
app = modal.App("write-myself2-compare-v4-v5-b200")
image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.12")
    .apt_install("curl")
    .run_commands(
        f"curl --fail --location --retry 3 https://codeload.github.com/NVIDIA/cutlass/tar.gz/{CUTLASS_COMMIT} -o /tmp/cutlass.tar.gz",
        "mkdir -p /opt/cutlass",
        "tar -xzf /tmp/cutlass.tar.gz -C /opt/cutlass --strip-components=1",
        "rm /tmp/cutlass.tar.gz",
    )
    .add_local_file(SOURCE_DIR / "v4.cu", "/root/user_v4.cu")
    .add_local_file(SOURCE_DIR / "v5.cu", "/root/user_v5.cu")
    .add_local_file(CUDA_DIR / "benchmark_v6.cu", "/root/benchmark_original.cu")
)


@app.function(image=image, gpu="B200", timeout=1800)
def compare(m: int, n: int, k: int, pairs: int) -> dict:
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

    run(["nvidia-smi"])
    run(["nvcc", "--version"])
    original = Path("/root/benchmark_original.cu").read_text()
    for version in (4, 5):
        Path(f"/root/variant_v{version}.cu").write_text(
            f'#include "user_v{version}.cu"\nnamespace mojo_v6 = v{version};\n'
        )
        source = original.replace(
            '#include "v6.cu"', f'#include "variant_v{version}.cu"', 1
        ).replace("CUDA v6:", f"CuTe v{version}:")
        Path(f"/root/benchmark_v{version}.cu").write_text(source)
        run([
            "nvcc", "-std=c++17", "-O3", "-lineinfo", "-arch=sm_100a",
            "-I/opt/cutlass/include", "-DMOJO_V6_MMA_N=256", "-Xptxas=-v",
            f"/root/benchmark_v{version}.cu", "-o", f"/root/benchmark_v{version}",
            "-L/usr/local/cuda/lib64/stubs", "-lcuda", "-lcublas",
        ])

    # Deterministic inputs, identical validation/cuBLAS/timing logic for both.
    # Alternate which kernel goes first, balancing slow drift across each pair.
    telemetry = ["nvidia-smi", "--query-gpu=uuid,name,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu", "--format=csv"]
    runs = {"v4": [], "v5": []}
    paired_results = []
    for pair in range(pairs):
        order = ("v4", "v5") if pair % 2 == 0 else ("v5", "v4")
        print(f"PAIR {pair + 1}/{pairs}: {' / '.join(order)}", flush=True)
        run(telemetry)
        pair_runs = {}
        for variant in order:
            output = run([f"/root/benchmark_{variant}", str(m), str(n), str(k)], 300)
            result = next(
                json.loads(line.removeprefix("RESULT_JSON="))
                for line in output.splitlines() if line.startswith("RESULT_JSON=")
            )
            runs[variant].append(result)
            pair_runs[variant] = result
        run(telemetry)
        paired_results.append({
            "pair": pair + 1, "order": order,
            "v4_ms": pair_runs["v4"]["custom_ms"],
            "v5_ms": pair_runs["v5"]["custom_ms"],
            "v5_latency_change_percent": (
                pair_runs["v5"]["custom_ms"] / pair_runs["v4"]["custom_ms"] - 1
            ) * 100,
        })

    medians = {name: statistics.median(r["custom_ms"] for r in values)
               for name, values in runs.items()}
    return {
        "m": m, "n": n, "k": k, "mma_n": 256, "pairs": pairs,
        "v4_ms": medians["v4"], "v5_ms": medians["v5"],
        "v5_latency_change_percent": (medians["v5"] / medians["v4"] - 1) * 100,
        "paired_latency_change_median_percent": statistics.median(
            pair["v5_latency_change_percent"] for pair in paired_results
        ),
        "paired_results": paired_results, "runs": runs,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "cuda_image": CUDA_IMAGE, "cutlass_commit": CUTLASS_COMMIT,
        "timing": "same GPU; balanced AB/BA pairs; unchanged benchmark_v6: deterministic inputs, correctness before/after timing, 100 direct warmups, 50 alternating warmup graph replays each, 100 GEMMs/replay, CUDA events, 7 alternating custom/cuBLAS rounds; median of run medians",
        "clocks_locked": False,
        "source_sha256": {name: hashlib.sha256(Path("/root", name).read_bytes()).hexdigest()
                          for name in ("user_v4.cu", "user_v5.cu", "benchmark_original.cu",
                                       "variant_v4.cu", "variant_v5.cu",
                                       "benchmark_v4.cu", "benchmark_v5.cu")},
        "logs": logs,
    }


@app.local_entrypoint()
def main(m: int = 4096, n: int = 4096, k: int = 4096, pairs: int = 6):
    import hashlib
    import json

    if min(m, n, k) <= 0 or m % 256 or n % 256 or k % 64:
        raise ValueError("Required: positive M,N,K; M % 256 = N % 256 = K % 64 = 0")
    if not 2 <= pairs <= 10 or pairs % 2:
        raise ValueError("pairs must be an even integer between 2 and 10")
    result = compare.remote(m, n, k, pairs)
    for path, name in ((SOURCE_DIR / "v4.cu", "user_v4.cu"),
                       (SOURCE_DIR / "v5.cu", "user_v5.cu"),
                       (CUDA_DIR / "benchmark_v6.cu", "benchmark_original.cu")):
        if hashlib.sha256(path.read_bytes()).hexdigest() != result["source_sha256"][name]:
            raise RuntimeError(f"Source changed during comparison: {path}")
    output = SOURCE_DIR / f"results_compare_v4_v5_b200_{m}x{n}x{k}.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {output}")
    print(f"v4 {result['v4_ms']:.6f} ms; v5 {result['v5_ms']:.6f} ms; "
          f"v5 latency change {result['v5_latency_change_percent']:+.2f}%; "
          f"median paired change {result['paired_latency_change_median_percent']:+.2f}%")
