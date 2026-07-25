import numpy as np
f32=np.float32; f64=np.float64
def vfma(a,b,c): return (f64(a)*f64(b)+f64(c)).astype(f32)

base='/home/julian/code/cpp/cajeta-xgboost/tools/expf/'
table=np.load(base+'ex2_table.npy').view(np.float32)   # ex2.approx(i/2^23), i in [0,2^23)
sweep=np.load(base+'expf_sweep.npy').view(np.float32)
N=len(sweep); TN=len(table)
step32=f32(f64(180.0)/(N-1))
i=np.arange(N,dtype=f64)
x=((i.astype(f32).astype(f64)*f64(step32))+f64(-90.0)).astype(f32)   # exact fmaf grid
dev=sweep
m=(x>-87.3)&(x<88.7)
xs=x[m].astype(f32); ds=dev[m]

L2E_HI=f32(1.442695021629333496)
# L2E_LO: low part so hi+lo ~ log2(e). log2e=1.4426950408889634; hi=1.4426950216...; lo~1.925e-8
L2E_LO=f32(1.4426950408889634-1.442695021629333496)   # ~1.925e-8

def ex2_table_approx(y):
    # ex2.approx(y) = ldexp(ex2.approx(frac(y)), floor(y)), frac in [0,1)
    yf=y.astype(f64)
    n=np.floor(yf).astype(np.int64)
    fr=(yf-n)                      # in [0,1)
    idx=np.rint(fr*TN).astype(np.int64)
    idx=np.clip(idx,0,TN-1)
    base_v=table[idx].astype(f64)
    # idx==TN would be frac->1; handled by clip, but rint could give TN
    return np.ldexp(base_v, n).astype(f32)

def R_single(x): return x*f32(1.442695041)
def R_fma2(x):   return vfma(x, L2E_HI, (x*L2E_LO).astype(f32))
def R_fma2b(x):
    t=vfma(x,L2E_HI,f32(0.0)); return vfma(x,L2E_LO,t)

def bits(a): return a.view(np.int32).astype(np.int64)
for name,R in [('single',R_single),('fma2',R_fma2),('fma2b',R_fma2b)]:
    red=R(xs)
    mdl=ex2_table_approx(red)
    mm=int((bits(mdl)!=bits(ds)).sum())
    # also try floor-index and +1 for the table lookup rounding
    print(f'{name:6s}: mismatches={mm}/{len(ds)}  ({100*mm/len(ds):.2f}%)')

# diagnostic: for fma2, distribution of bit error
red=R_fma2(xs); mdl=ex2_table_approx(red)
d=bits(mdl)-bits(ds)
u,c=np.unique(np.clip(d,-5,5),return_counts=True)
print('fma2 bit-error hist (clipped +/-5):',dict(zip(u.tolist(),c.tolist())))
