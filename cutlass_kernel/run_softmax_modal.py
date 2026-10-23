"""BF16 handwritten CuTe softmax vs PyTorch on B200; 4096 x 4096 by default."""
from pathlib import Path
import modal

SOURCE_DIR = Path(__file__).resolve().parent
CUDA_IMAGE = "nvidia/cuda:12.8.1-devel-ubuntu24.04"
CUTLASS_COMMIT = "f3fde58372d33e9a5650ba7b80fc48b3b49d40c8"
app = modal.App("cute-softmax-bf16-b200")
image = (modal.Image.from_registry(CUDA_IMAGE, add_python="3.12")
         .apt_install("curl")
         .run_commands(
             f"curl --fail --location --retry 3 https://codeload.github.com/NVIDIA/cutlass/tar.gz/{CUTLASS_COMMIT} -o /tmp/cutlass.tar.gz",
             "mkdir -p /opt/cutlass",
             "tar -xzf /tmp/cutlass.tar.gz -C /opt/cutlass --strip-components=1",
             "rm /tmp/cutlass.tar.gz")
         .pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cu128"))
for name in ["softmax.cu", "softmax_binding.cu", "softmax_check.cu"]:
    image = image.add_local_file(SOURCE_DIR / name, f"/root/{name}")


@app.function(image=image, gpu="B200", timeout=1800)
def benchmark(rows: int = 4096, cols: int = 4096, sanitize: bool = False) -> dict:
    import ctypes
    import datetime
    import hashlib
    import json
    import statistics
    import subprocess
    import torch

    logs = []
    def run(command, timeout=300):
        p = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, timeout=timeout)
        print(p.stdout, flush=True)
        logs.append({"command": command, "output": p.stdout})
        p.check_returncode()
        return p.stdout
    run(["nvidia-smi"])
    run(["nvcc", "--version"])
    common = ["nvcc", "-std=c++17", "-O3", "-DNDEBUG", "-lineinfo", "-arch=sm_100a",
              "-I/opt/cutlass/include", "/root/softmax_binding.cu"]
    run(common + ["--shared", "-Xcompiler=-fPIC", "-Xptxas=-v", "-o", "/root/softmax.so"])
    run(common + ["--ptx", "-o", "/root/softmax.ptx"])
    run(["nvcc", "-std=c++17", "-O3", "-arch=sm_100a", "/root/softmax_check.cu",
         "-L/root", "-l:softmax.so", "-Xlinker=-rpath=/root", "-o", "/root/softmax_check"])
    run(["/root/softmax_check"])
    if sanitize:
        for tool in ["memcheck", "synccheck", "racecheck"]:
            run(["compute-sanitizer", "--tool", tool, "--error-exitcode", "1", "/root/softmax_check"])
    lib = ctypes.CDLL("/root/softmax.so")
    lib.softmax_bf16.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
    lib.softmax_bf16.restype = ctypes.c_int
    def launch(x, y, threads=0):
        status = lib.softmax_bf16(x.data_ptr(), y.data_ptr(), x.shape[0], x.shape[1],
                                 torch.cuda.current_stream().cuda_stream, threads)
        if status:
            raise RuntimeError(f"softmax CUDA status={status}")

    torch.manual_seed(20261007)
    correctness = []
    def validate(x, label, threads=0, unaligned_output=False):
        y = (torch.empty(x.numel() + 1, device="cuda", dtype=torch.bfloat16)[1:].view_as(x)
             if unaligned_output else torch.empty_like(x))
        launch(x, y, threads)
        # FP64 reference uses the actual rounded BF16 input.
        ref64 = torch.softmax(x.double(), dim=-1)
        ref = ref64.bfloat16()
        torch.testing.assert_close(y, ref, rtol=0.008, atol=2e-7, equal_nan=True)
        finite_rows = torch.isfinite(ref64).all(dim=-1)
        if finite_rows.any():
            row_error = (y[finite_rows].float().sum(-1) - 1).abs().max().item()
            assert row_error < 0.008, row_error
            assert (y[finite_rows] >= 0).all().item()
        finite = torch.isfinite(ref)
        error = (y[finite].float() - ref[finite].float()).abs().max().item() if finite.any() else 0
        record = {"case": label, "shape": list(x.shape), "threads": threads,
                  "max_abs_error_vs_fp64_bf16": error}
        correctness.append(record)
        print("PASS " + json.dumps(record), flush=True)
    for m, n in [(3, 1), (7, 17), (5, 129), (9, 1024), (11, 2048), (7, 4095),
                 (13, 4096), (3, 4097), (3, 8192), (2, 32768), (2, 32769)]:
        validate(torch.randn((m, n), device="cuda", dtype=torch.bfloat16) * 3, "random")
    for threads in [64, 128, 256]:
        validate(torch.randn((13, 4096), device="cuda", dtype=torch.bfloat16) * 3,
                 "4096 configuration", threads)
    x_unaligned = torch.randn(7 * 4096 + 1, device="cuda", dtype=torch.bfloat16)[1:].view(7, 4096)
    validate(x_unaligned, "unaligned input/output", unaligned_output=True)
    validate(torch.full((3, 4096), 10000., device="cuda", dtype=torch.bfloat16), "constant large logits")
    validate(torch.randn((3, 4096), device="cuda", dtype=torch.bfloat16) * 1000, "large logits")
    x = torch.randn((4, 4096), device="cuda", dtype=torch.bfloat16)
    x[0].fill_(-float("inf")); x[1, ::2] = -float("inf")
    x[2, 0] = float("inf"); x[3, 0] = float("nan")
    validate(x, "masked and nonfinite rows")

    x = torch.randn((rows, cols), device="cuda", dtype=torch.bfloat16)
    y = torch.empty_like(x)
    reference = torch.empty_like(x)
    validate(x, "benchmark input")
    iterations, rounds = 100, 9
    def measure(launches):
        graphs = {}
        for name, fn in launches.items():
            for i in range(20): fn(i)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for i in range(iterations): fn(i)
            graph.replay()
            torch.cuda.synchronize()
            graphs[name] = graph
        samples = {name: [] for name in graphs}
        names = list(graphs)
        for round_index in range(rounds):
            order = names[round_index % len(names):] + names[:round_index % len(names)]
            for name in order:
                start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record(); graphs[name].replay(); stop.record(); stop.synchronize()
                samples[name].append(start.elapsed_time(stop) / iterations)
        measured = {}
        for name, values in samples.items():
            ms = statistics.median(values)
            measured[name] = {"ms": ms, "us": ms * 1000, "samples_ms": values,
                              "effective_GBps": rows * cols * 4 / ms / 1e6}
        for t in [64, 128, 256]:
            measured[f"cute_{t}"]["speedup_vs_pytorch"] = measured["pytorch"]["ms"] / measured[f"cute_{t}"]["ms"]
        return measured
    launches = {f"cute_{t}": (lambda i, t=t: launch(x, y, t)) for t in [64, 128, 256]}
    launches["pytorch"] = lambda i: torch.softmax(x, dim=-1, out=reference)
    results = measure(launches)
    for t in [64, 128, 256]:
        launch(x, y, t)
        torch.testing.assert_close(y, reference, rtol=0.008, atol=2e-7)
        results[f"cute_{t}"]["speedup_vs_pytorch"] = results["pytorch"]["ms"] / results[f"cute_{t}"]["ms"]
    best = min([64, 128, 256], key=lambda t: results[f"cute_{t}"]["ms"])
    # Eight rotating input/output buffers exceed B200 L2 for the requested 4096^2 case.
    ring_size = 8
    input_ring = list(torch.randn((ring_size, rows, cols), device="cuda", dtype=torch.bfloat16).unbind(0))
    output_ring = list(torch.empty((ring_size, rows, cols), device="cuda", dtype=torch.bfloat16).unbind(0))
    ring_launches = {f"cute_{t}": (lambda i, t=t: launch(input_ring[i % ring_size], output_ring[i % ring_size], t))
                     for t in [64, 128, 256]}
    ring_launches["pytorch"] = lambda i: torch.softmax(input_ring[i % ring_size], dim=-1, out=output_ring[i % ring_size])
    streaming_results = measure(ring_launches)
    result = {
        "rows": rows, "cols": cols, "dtype": "bfloat16", "gpu": torch.cuda.get_device_name(),
        "torch_version": str(torch.__version__), "torch_cuda": torch.version.cuda,
        "cuda_image": CUDA_IMAGE, "cutlass_commit": CUTLASS_COMMIT,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "iterations": iterations, "rounds": rounds, "best_threads": best,
        "timing": "CUDA Graph 100 operations/replay; CUDA events; 9 rounds with rotating order; median",
        "working_set": "cached: same buffers; streaming: 8 rotating input/output pairs (512 MiB at 4096^2)",
        "results": results, "streaming_results": streaming_results,
        "sanitize": sanitize, "correctness": correctness, "logs": logs,
        "source_sha256": {n: hashlib.sha256(Path('/root', n).read_bytes()).hexdigest()
                          for n in ["softmax.cu", "softmax_binding.cu", "softmax_check.cu"]},
        "ptx": Path("/root/softmax.ptx").read_text(),
    }
    print(json.dumps({"best_threads": best, "results": results, "streaming_results": streaming_results}, indent=2), flush=True)
    return result


@app.local_entrypoint()
def main(rows: int = 4096, cols: int = 4096, sanitize: bool = False):
    import json
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")
    result = benchmark.remote(rows, cols, sanitize)
    (SOURCE_DIR / "softmax.sm100a.ptx").write_text(result.pop("ptx"))
    path = SOURCE_DIR / f"results_softmax_bf16_b200_{rows}x{cols}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {path}; best threads={result['best_threads']}")
