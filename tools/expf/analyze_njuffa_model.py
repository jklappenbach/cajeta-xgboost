import numpy as np
f32=np.float32; f64=np.float64
def fma(a,b,c):  # exact single-rounding f32 fma (scalar arrays)
    return (f64(a)*f64(b)+f64(c)).astype(f32)

base='/home/julian/code/cpp/cajeta-xgboost/tools/expf/'
sweep=np.load(base+'expf_sweep.npy').view(np.float32)
N=len(sweep)
step32=f32(f64(180.0)/(N-1))
i=np.arange(N,dtype=f64)
x=((i.astype(f32).astype(f64)*f64(step32))+f64(-90.0)).astype(f32)
dev=sweep

# njuffa accurate expf (libdevice-style): magic-number rint, 2-part ln2, deg-6 poly
MAGIC=f32(12582912.0)   # 1.5*2^23
L2E=f32(1.442695041)
LN2HI=f32(6.93145752e-1)
LN2LO=f32(1.42860677e-6)
# candidate coefficient sets (njuffa has posted a few; test several)
COEFFS={
 'A':[1.37805939e-3,8.37312452e-3,4.16695364e-2,1.66664720e-1,4.99999851e-1,1.0,1.0],
 'B':[1.378059e-3,8.373104e-3,4.166999e-2,1.666665e-1,4.999999e-1,1.0,1.0],
 'C':[1.3653896e-3,8.3736268e-3,4.1665081e-2,1.6666277e-1,5.0000000e-1,1.0000000e+0,1.0000000e+0],
}
def expf_nj(xv, c):
    j=(fma(L2E,xv,MAGIC)); j=(j-MAGIC).astype(f32)   # rint(a*log2e)
    f=fma(j,f32(-LN2HI),xv); f=fma(j,f32(-LN2LO),f)
    r=np.full(len(xv),f32(c[0]),dtype=f32)
    for k in range(1,7): r=fma(r,f,f32(c[k]))
    ii=j.astype(np.int64)
    return np.ldexp(r.astype(f64),ii).astype(f32)

def bits(a): return a.view(np.int32).astype(np.int64)
m=(x>-87.3)&(x<88.7)
xs=x[m].astype(f32); ds=dev[m]
for name,c in COEFFS.items():
    mdl=expf_nj(xs,c)
    d=bits(mdl)-bits(ds)
    mm=int((d!=0).sum())
    print(f'coeffs {name}: mismatch {mm}/{len(ds)} ({100*mm/len(ds):.3f}%)  maxULP={int(np.abs(d).max())}')
