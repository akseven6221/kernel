"""Reproducible B200 CUDA/PTX attention experiments; artifacts are returned locally."""
from pathlib import Path
import modal
ROOT = Path(__file__).resolve().parent
app = modal.App("fa4-cuda-ptx")
image = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
         .pip_install("torch==2.10.0", "flash-attn-4==4.0.0b35", "nvidia-cutlass-dsl==4.8.0", "quack-kernels==0.6.5", "apache-tvm-ffi==0.1.14.post1", "numpy", "ninja")
         .add_local_dir(ROOT, "/root/fa4", ignore=["runs/**", "**/results/**", "**/profile/**", "**/results.json", "**/__pycache__/**", "**/*.log", "reference/chapter.html"]))
@app.function(image=image, gpu="B200", timeout=3600)
def experiment(versions: str, mode: str, seq: int = 4096, causal: bool = False, run_timeout: int = 900):
    import subprocess, json, sys, datetime, hashlib, time, threading
    from pathlib import Path
    out = Path("/tmp/artifacts"); out.mkdir(exist_ok=True)
    def run(cmd, required=True, timeout=900):
        print("$ " + " ".join(cmd), flush=True)
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
            lines = []
            timer = threading.Timer(timeout, process.kill); timer.start()
            try:
                for line in process.stdout:
                    if "--import" not in cmd: print(line[:10000], end="", flush=True)
                    lines.append(line)
                process.wait()
            finally: timer.cancel()
        p = subprocess.CompletedProcess(cmd, process.returncode, "".join(lines))
        if required: p.check_returncode()
        return p
    env = run(["nvidia-smi"]).stdout + run(["nvcc", "--version"]).stdout
    out.joinpath("environment.txt").write_text(env)
    out.joinpath("run.json").write_text(json.dumps({"timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "versions": versions, "mode": mode, "profile_seq": seq, "profile_causal": causal, "cuda_image": "nvidia/cuda:12.8.1-devel-ubuntu24.04"}, indent=2))
    out.joinpath("pip-freeze.txt").write_text(run([sys.executable, "-m", "pip", "freeze"]).stdout)
    for version in versions.split(","):
        source = Path("/root/fa4") / version / "kernel.cu"
        if source.exists():
            dest = out / version; dest.mkdir(exist_ok=True); dest.joinpath("kernel.cu").write_bytes(source.read_bytes())
    out.joinpath("benchmark.py").write_bytes(Path("/root/fa4/benchmark.py").read_bytes())
    out.joinpath("run_modal.py").write_bytes(Path("/root/fa4/run_modal.py").read_bytes())
    if mode == "setup":
        run([sys.executable,"-c","import torch, importlib.metadata as m; from flash_attn.cute import flash_attn_func; print(torch.__version__, m.version('flash-attn-4')); print(torch.cuda.get_device_name())"])
    else:
        p = run([sys.executable,"/root/fa4/benchmark.py","--versions",versions,"--mode",("prepare" if mode in ["profile", "sanitize"] else mode),"--output",str(out)], required=False, timeout=run_timeout)
        out.joinpath("execution.log").write_text(p.stdout)
        out.joinpath("exit_code.txt").write_text(str(p.returncode))
        if p.returncode == 0 and mode == "profile":
            for version in versions.split(",") + ["official"]:
                dest = out / version; dest.mkdir(exist_ok=True)
                cmd = ["ncu", "--set", "full", "--clock-control", "none", "--target-processes", "all", "--launch-count", "1", "--export", str(dest / "profile"), "--force-overwrite"]
                if version != "official": cmd += ["--kernel-name-base", "demangled", "--kernel-name", "regex:fa4::attention"]
                else: cmd += ["--nvtx", "--nvtx-include", "official/"]
                cmd += [sys.executable, "/root/fa4/benchmark.py", "--versions", version, "--mode", ("official" if version == "official" else "probe"), "--output", str(out)]
                cmd += ["--seq", str(seq)]
                if causal: cmd.append("--causal")
                r = run(cmd, required=False, timeout=run_timeout); dest.joinpath("ncu.log").write_text(r.stdout); dest.joinpath("ncu.exit_code").write_text(str(r.returncode))
                if dest.joinpath("profile.ncu-rep").exists():
                    for page in ["details", "raw", "source"]:
                        r = run(["ncu", "--import", str(dest / "profile.ncu-rep"), "--page", page, "--csv"] if page == "raw" else ["ncu", "--import", str(dest / "profile.ncu-rep"), "--page", page], required=False)
                        dest.joinpath("ncu_" + page + (".csv" if page == "raw" else ".txt")).write_text(r.stdout)
        if p.returncode == 0 and mode == "sanitize":
            for version in versions.split(","):
                for tool in ["memcheck", "synccheck", "racecheck"]:
                    for causal in [False, True]:
                        cmd = ["compute-sanitizer", "--tool", tool, "--error-exitcode", "1", sys.executable, "/root/fa4/benchmark.py", "--versions", version, "--mode", "sanitize", "--seq", "511", "--output", str(out)]
                        if causal: cmd.append("--causal")
                        r = run(cmd, required=False, timeout=run_timeout)
                        name = tool + ("_causal" if causal else "_noncausal")
                        out.joinpath(version, name + ".txt").write_text(r.stdout)
                        out.joinpath(version, name + ".exit_code").write_text(str(r.returncode))
    import io, zipfile
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for p in out.rglob("*"):
            if p.is_file() and p.suffix in [".txt", ".json", ".cu", ".py", ".ptx", ".csv", ".ncu-rep", ".log", ".exit_code", ".cubin"]:
                archive.writestr(str(p.relative_to(out)), p.read_bytes())
    print(f"Returning {len(buffer.getvalue())} compressed artifact bytes", flush=True)
    return buffer.getvalue()
@app.local_entrypoint()
def main(versions: str = "v1", mode: str = "benchmark", label: str = "", seq: int = 4096, causal: bool = False, run_timeout: int = 900):
    import datetime
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = ROOT / "runs" / (label or f"{stamp}_{versions.replace(',', '_')}_{mode}")
    dest.mkdir(parents=True, exist_ok=False)
    try:
        results = experiment.remote(versions, mode, seq, causal, run_timeout)
        if isinstance(results, dict):
            for name, data in results.items():
                p = dest / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(data)
        else:
            import io, zipfile
            with zipfile.ZipFile(io.BytesIO(results)) as archive: archive.extractall(dest)
    except Exception as e:
        dest.joinpath("FAILED.txt").write_text(repr(e)); raise
    print(f"Saved {dest}")
    failures=[]
    for p in [dest/"exit_code.txt", *dest.rglob("*.exit_code")]:
        if p.exists() and p.read_text().strip()!="0": failures.append(str(p.relative_to(dest)))
    if mode=="profile":
        for version in versions.split(",")+["official"]:
            if not (dest/version/"profile.ncu-rep").exists(): failures.append(version+"/missing profile.ncu-rep")
    if failures: raise RuntimeError(f"Failed checks: {failures}; artifacts saved in {dest}")
