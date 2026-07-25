import numpy as np
f32=np.float32; f64=np.float64
def fma(a,b,c): return (f64(a)*f64(b)+f64(c)).astype(f32)
base='/home/julian/code/cpp/cajeta-xgboost/tools/expf/'
sweep=np.load(base+'expf_sweep.npy').view(np.float32)
N=len(sweep); step32=f32(f64(180.0)/(N-1))
i=np.arange(N,dtype=f64)
x=((i.astype(f32).astype(f64)*f64(step32))+f64(-90.0)).astype(f32)
m=(x>-87.3)&(x<88.7); xall=x[m].astype(f32); dall=sweep[m]
sub=np.linspace(0,len(xall)-1,120000).astype(np.int64)
xs=xall[sub]; ds=dall[sub]
def bits(a): return a.view(np.int32).astype(np.int64)
dsb=bits(ds)

MAGIC=f32(12582912.0)
# params: [L2E, LN2HI, LN2LO, c0..c6]
P=[f32(1.442695041),f32(6.93145752e-1),f32(1.42860677e-6),
   f32(1.37805939e-3),f32(8.37312452e-3),f32(4.16695364e-2),f32(1.66664720e-1),f32(4.99999851e-1),f32(1.0),f32(1.0)]
def model(P,xv):
    L2E,LN2HI,LN2LO=P[0],P[1],P[2]; c=P[3:]
    j=(fma(L2E,xv,MAGIC)-MAGIC).astype(f32)
    f=fma(j,f32(-LN2HI),xv); f=fma(j,f32(-LN2LO),f)
    r=np.full(len(xv),f32(c[0]),dtype=f32)
    for k in range(1,7): r=fma(r,f,f32(c[k]))
    return np.ldexp(r.astype(f64),j.astype(np.int64)).astype(f32)
def mism(P): return int((bits(model(P,xs))!=dsb).sum())
def neigh(v,s):
    b=np.float32(v).view(np.int32); return [np.int32(b+d).view(np.float32) for d in range(-s,s+1)]
best=list(P); bm=mism(best); print('start',bm,f'({100*bm/len(ds):.3f}%)')
for it in range(12):
    imp=False
    for p in range(len(best)):
        span=3 if p<3 else 4
        base_v=best[p]
        for cand in neigh(base_v,span):
            t=list(best); t[p]=cand; mm=mism(t)
            if mm<bm: bm=mm; best=t; imp=True
    print(f'it{it}: {bm} ({100*bm/len(ds):.3f}%)')
    if bm==0 or not imp: break
# validate on full domain
fullb=(bits(model(best,xall))!=bits(dall)).sum()
print('FULL-domain mismatch:',int(fullb),'/',len(xall),f'({100*fullb/len(xall):.4f}%)')
print('best P bits:',[int(np.float32(v).view(np.int32)) for v in best])
print('best P vals:',[f'{float(v):.9g}' for v in best])
