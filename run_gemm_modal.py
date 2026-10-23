"""Compile and run kernel/gemm1.cu (Blackwell sm_100a BF16 GEMM) on a Modal B200 GPU.

Usage:
    modal run kernel/run_gemm_modal.py              # correctness test, M=N=K=4096
    modal run kernel/run_gemm_modal.py --benchmark  # 100 warmup + 100 timed runs
    modal run kernel/run_gemm_modal.py --profile    # ncu --set full on both kernel configs
"""

import modal

app = modal.App("blackwell-gemm1")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .add_local_file("kernel/gemm1.cu", remote_path="/root/gemm1.cu")
)

BUILD_CMD = [
    "nvcc",
    "-std=c++17",
    "-O3",
    "-arch=sm_100a",
    "/root/gemm1.cu",
    "-o",
    "/root/blackwell_matmul",
    "-L/usr/local/cuda/lib64/stubs",
    "-lcuda",
]


@app.function(image=image, gpu="B200", timeout=1800)
def compile_and_run(benchmark: bool = False) -> None:
    import subprocess

    subprocess.run(["nvidia-smi"], check=True)
    subprocess.run(BUILD_CMD, check=True)
    print("=== nvcc build OK ===", flush=True)

    cmd = ["/root/blackwell_matmul"]
    if benchmark:
        cmd.append("--benchmark")
    subprocess.run(cmd, check=True)


@app.function(image=image, gpu="B200", timeout=1800)
def profile() -> dict[str, bytes]:
    import shutil
    import subprocess

    subprocess.run(BUILD_CMD, check=True)
    print("=== nvcc build OK ===", flush=True)

    ncu = shutil.which("ncu")
    if ncu is None:
        raise RuntimeError("ncu (Nsight Compute CLI) not found in image")

    # Two kernel configs in the binary: correctness <N=128, 8 stages> and
    # benchmark <N=256, 7 stages>. Profile one launch of each.
    runs = [
        ("correctness", ["/root/blackwell_matmul"], ["--launch-count", "1"]),
        # 100 warmup + 100 timed launches; profile one warm timed launch.
        ("benchmark", ["/root/blackwell_matmul", "--benchmark"], ["--launch-skip", "110", "--launch-count", "1"]),
    ]
    reports: dict[str, bytes] = {}
    for name, cmd, launch_sel in runs:
        out = f"/root/profile_{name}"
        print(f"\n########## ncu: {name} config ##########", flush=True)
        subprocess.run(
            [
                ncu,
                "--kernel-name-base",
                "demangled",
                "--kernel-name",
                "regex:blackwell",
                *launch_sel,
                "--set",
                "full",
                "--clock-control",
                "none",
                "--export",
                out,
                "--force-overwrite",
                *cmd,
            ],
            check=True,
        )
        with open(out + ".ncu-rep", "rb") as f:
            reports[name] = f.read()
        # --export suppresses the stdout report; re-import to print the
        # classic ncu details page.
        subprocess.run([ncu, "--import", out + ".ncu-rep", "--page", "details"], check=True)
    return reports


@app.local_entrypoint()
def main(benchmark: bool = False, profile: bool = False) -> None:
    if profile:
        reports = globals()["profile"].remote()
        for name, data in reports.items():
            path = f"kernel/gemm1_profile_{name}.ncu-rep"
            with open(path, "wb") as f:
                f.write(data)
            print(f"saved {path} ({len(data)} bytes)")
    else:
        compile_and_run.remote(benchmark)
