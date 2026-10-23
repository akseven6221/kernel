"""CUDA/PTX FA4 validation, same-device paired graph timing and profiler target."""
import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent

def command(args, logfile, timeout=300):
    p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
    logfile.write_text('$ ' + ' '.join(args) + '\n' + p.stdout)
    print(p.stdout if len(p.stdout)<12000 else f"Saved {logfile}: {len(p.stdout)} bytes", flush=True)
    if p.returncode: raise RuntimeError(f'command failed ({p.returncode}): {args}; see {logfile}')
    return p.stdout

class Kernel:
    def __init__(self, version, out, compile=True):
        self.version = version
        self.dest = out / version
        self.dest.mkdir(parents=True, exist_ok=True)
        source = ROOT / version / 'kernel.cu'
        self.so = Path('/tmp') / f'fa4_{version}.so'
        if compile:
            command(['nvcc','-std=c++17','-O3','-lineinfo','-arch=sm_100a','--shared','-Xcompiler=-fPIC',
                     '--keep','--keep-dir',str(self.dest),'-Xptxas=-v',str(source),'-o',str(self.so),
                     '-L/usr/local/cuda/lib64/stubs','-lcuda'],self.dest/'build.txt')
            command(['cuobjdump','--dump-sass',str(self.so)],self.dest/'sass.txt')
        self.lib = ctypes.CDLL(str(self.so))
        P=ctypes.c_void_p; I=ctypes.c_int
        self.lib.fa4_create.argtypes=[ctypes.POINTER(P),P,P,P,I,I,I]; self.lib.fa4_create.restype=I
        self.lib.fa4_launch.argtypes=[P,P,I,P]; self.lib.fa4_launch.restype=I
        self.lib.fa4_destroy.argtypes=[P]
        self.sha = hashlib.sha256(source.read_bytes()).hexdigest()
    def bind(self,q,k,v,causal=False):
        import torch
        assert q.shape==k.shape==v.shape and q.dtype==torch.float16 and q.is_contiguous() and q.shape[-1]==128
        b,n,h,d=q.shape; ctx=ctypes.c_void_p()
        rc=self.lib.fa4_create(ctypes.byref(ctx),q.data_ptr(),k.data_ptr(),v.data_ptr(),b,n,h)
        if rc: raise RuntimeError(f'fa4_create error {rc}')
        o=torch.empty_like(q)
        def launch():
            rc=self.lib.fa4_launch(ctx,o.data_ptr(),int(causal),torch.cuda.current_stream().cuda_stream)
            if rc: raise RuntimeError(f'fa4_launch error {rc}')
            return o
        return launch,o,ctx

def reference(q,k,v,causal):
    import torch
    qh=q.float().transpose(1,2); kh=k.float().transpose(1,2); vh=v.float().transpose(1,2)
    chunks=[]; n=q.shape[1]
    for start in range(0,n,128):
        s=qh[:,:,start:start+128]@kh.transpose(-1,-2)/128**0.5
        if causal:
            mask=torch.arange(n,device=q.device)[None,:]>torch.arange(start,min(start+128,n),device=q.device)[:,None]
            s.masked_fill_(mask,float('-inf'))
        chunks.append(torch.softmax(s,-1)@vh)
    return torch.cat(chunks,2).transpose(1,2)

def errors(o,ref):
    import torch
    delta=o.float()-ref.float()
    return dict(finite=bool(torch.isfinite(o).all()),max_abs=float(delta.abs().max()),rmse=float(delta.square().mean().sqrt()),
                relative_l2=float(delta.norm()/ref.float().norm().clamp_min(1e-12)),
                passed=bool(torch.allclose(o.float(),ref.float(),atol=0.003,rtol=0.003)))

def graphs(functions):
    import torch
    stream=torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
    gs={}
    with torch.cuda.stream(stream):
        for name,fn in functions.items():
            for _ in range(3): fn()
            stream.synchronize(); g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g,stream=stream):
                for _ in range(20): fn()
            gs[name]=g
    torch.cuda.current_stream().wait_stream(stream); torch.cuda.synchronize()
    for _ in range(10):
        for g in gs.values():g.replay()
    torch.cuda.synchronize(); samples={name:[] for name in gs}
    for r in range(9):
        names=list(gs); names=names if r%2==0 else names[::-1]
        for name in names:
            start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
            start.record();gs[name].replay();end.record();end.synchronize()
            samples[name].append(start.elapsed_time(end)*1000/20)
    return {k:dict(median_us=statistics.median(v),samples_us=v,min_us=min(v),max_us=max(v)) for k,v in samples.items()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--versions',default='v1');p.add_argument('--mode',default='benchmark');p.add_argument('--output',type=Path,default=Path('/tmp/artifacts'));p.add_argument('--seq',type=int,default=4096);p.add_argument('--causal',action='store_true')
    args=p.parse_args(); out=args.output;out.mkdir(parents=True,exist_ok=True)
    import torch
    torch.set_grad_enabled(False); torch.backends.cuda.matmul.allow_tf32=False
    from flash_attn.cute import flash_attn_func as official_function
    def flash_attn_func(*args, **kwargs):
        result=official_function(*args, **kwargs)
        return result[0] if isinstance(result,tuple) else result
    torch.manual_seed(20261014)
    meta=dict(torch=torch.__version__,device=torch.cuda.get_device_name(),capability=torch.cuda.get_device_capability(),
              packages={k:importlib.metadata.version(k) for k in ['flash-attn-4','nvidia-cutlass-dsl','apache-tvm-ffi','quack-kernels']},
              timing='CUDA Graph, 20 nodes, 10 warmups, 9 alternating rounds; CUDA events; median',pid=os.getpid())
    import flash_attn.cute
    official_dir=out/'official'; official_dir.mkdir(exist_ok=True)
    package=Path(flash_attn.cute.__file__).parent
    meta['official_source_sha256']={}
    for name in ['interface.py','flash_fwd_sm100.py','softmax.py','utils.py']:
        data=(package/name).read_bytes(); (official_dir/name).write_bytes(data)
        meta['official_source_sha256'][name]=hashlib.sha256(data).hexdigest()
    (out/'metadata.json').write_text(json.dumps(meta,indent=2))
    kernels=[] if args.mode=='official' else [Kernel(v,out,compile=args.mode not in ['probe','sanitize']) for v in args.versions.split(',')]
    if args.mode=='prepare': return
    if args.mode in ['probe','official','sanitize']:
        n=args.seq
        q,k,v=[torch.randn(1,n,8,128,device='cuda',dtype=torch.float16) for _ in range(3)]
        if args.mode=='official':
            flash_attn_func(q,k,v,causal=args.causal);torch.cuda.synchronize()
            torch.cuda.nvtx.range_push('official');flash_attn_func(q,k,v,causal=args.causal);torch.cuda.nvtx.range_pop();torch.cuda.synchronize()
        else:
            launch,o,ctx=kernels[0].bind(q,k,v,args.causal); launch();torch.cuda.synchronize()
            if args.mode=='sanitize':
                ref=reference(q,k,v,args.causal); e=errors(o,ref);print(json.dumps(e));assert e['passed']
            kernels[0].lib.fa4_destroy(ctx)
        return
    results=dict(metadata=meta,correctness=[],benchmarks=[])
    def save(): (out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    # Includes non-multiple tile lengths and large logits, which stress rescaling.
    for n in [128,256,257,511]:
        for causal in [False,True]:
            for distribution in ['normal','zeros','large_logits','constant_large_logits']:
                q,k,v=[torch.randn(1,n,2,128,device='cuda',dtype=torch.float16) for _ in range(3)]
                if distribution=='zeros':q.zero_();k.zero_()
                if distribution=='large_logits':q.mul_(6);k.mul_(6)
                if distribution=='constant_large_logits':q.fill_(30000);k.fill_(30000)
                ref=reference(q,k,v,causal)
                official=flash_attn_func(q,k,v,causal=causal);torch.cuda.synchronize()
                official_err=errors(official,ref)
                for kernel in kernels:
                    launch,o,ctx=kernel.bind(q,k,v,causal);launch();torch.cuda.synchronize()
                    e=errors(o,ref);entry=dict(version=kernel.version,seq=n,causal=causal,distribution=distribution,error=e,official_error=official_err)
                    results['correctness'].append(entry);save();print(json.dumps(entry),flush=True)
                    kernel.lib.fa4_destroy(ctx)
                    if not e['passed']:raise AssertionError(entry)
    if args.mode=='check': return
    for b,n,h,causal in [(1,512,8,False),(1,1024,8,False),(1,4096,8,False),(1,8192,8,False),(1,4096,8,True),(1,8192,8,True),(2,4096,16,False)]:
        q,k,v=[torch.randn(b,n,h,128,device='cuda',dtype=torch.float16) for _ in range(3)]
        official=lambda:flash_attn_func(q,k,v,causal=causal)
        ref=official();torch.cuda.synchronize()
        funcs={'official_fa4':official}; contexts=[];shape_errors={}
        for kernel in kernels:
            launch,o,ctx=kernel.bind(q,k,v,causal);launch();torch.cuda.synchronize()
            e=errors(o,ref);assert e['passed'],e
            shape_errors[kernel.version]=e;funcs[kernel.version]=launch;contexts.append((kernel,ctx))
        timing=graphs(funcs);official_us=timing['official_fa4']['median_us']
        for name,t in timing.items():
            t['tflops']=4*b*h*n*n*128*(0.5 if causal else 1)/t['median_us']/1e6
            t['official_throughput_percent']=100*official_us/t['median_us']
        entry=dict(shape=[b,n,h,128],causal=causal,timing=timing,errors=shape_errors,source_sha256={x.version:x.sha for x in kernels})
        results['benchmarks'].append(entry);save();print(json.dumps(entry),flush=True)
        for kernel,ctx in contexts:kernel.lib.fa4_destroy(ctx)
    save()
if __name__=='__main__':main()
