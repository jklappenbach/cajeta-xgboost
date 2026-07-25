import numpy as np
f32=np.float32; f64=np.float64

def fmaf(a,b,c):  # exact single-rounding f32 FMA
    return np.float32(f64(a)*f64(b)+f64(c))
# vectorized f32 fma
def vfma(a,b,c):
    return (f64(a)*f64(b)+f64(c)).astype(f32)

sweep=np.load('/home/julian/code/cpp/cajeta-xgboost/tools/expf/expf_sweep.npy').view(np.float32)
N=len(sweep)
step32=f32(f64(180.0)/(N-1))

# full grid via exact fmaf(i,step,-90)
i=np.arange(N,dtype=f64)
x=( (i.astype(f32).astype(f64)*f64(step32)) + f64(-90.0) ).astype(f32)   # exact fmaf
dev=sweep

# domain XGBoost uses: sigmoid clamps to x<=88.7, and exp of very negative -> 0.
# focus where result is normal (x in ~[-87,88.7]); subsample for the search
m=(x> -87.3)&(x<88.7)
xs=x[m]; ds=dev[m]
sub=np.linspace(0,len(xs)-1,150000).astype(np.int64)
xs=xs[sub]; ds=ds[sub]

L2E=f32(1.4426950408889634)
def reduce_jf(xv, ln2hi, ln2lo):
    # j = rintf(x*L2E) ; f = fmaf(j,-ln2hi,x); f=fmaf(j,-ln2lo,f)
    t=(f64(xv)*f64(L2E))
    j=np.rint(t).astype(f64)          # rintf (round-half-even)
    jf=j.astype(f32).astype(f64)
    f=(( -jf*f64(ln2hi) )+f64(xv)).astype(f32)
    f=(( -jf*f64(ln2lo) )+f64(f)).astype(f32)
    return j.astype(np.int64), f

def eval_poly_horner(f, c):
    r=np.full(len(f), f32(c[-1]), dtype=f32)
    for k in range(len(c)-2,-1,-1):
        r=vfma(r,f,f32(c[k]))
    return r

def expf_model(xv, c, ln2hi, ln2lo):
    j,f=reduce_jf(xv, ln2hi, ln2lo)
    r=eval_poly_horner(f, c)
    out=np.ldexp(r.astype(f64), j).astype(f32)
    return out

def bits(a): return a.view(np.int32).astype(np.int64)
def mism(model):
    return int((bits(model)!=bits(ds)).sum())

# --- fit initial poly with a standard ln2 split ---
LN2HI=f32(0.693359375)      # common Cody-Waite hi (exact in f32: 0.693359375)
LN2LO=f32(-2.12194440e-4)   # so hi+lo ~ ln2 ; sign convention: ln2 = hi + lo
# try ln2 = hi + lo  => reduce uses f = x - j*(hi+lo). We coded -j*hi then -j*lo, so want hi+lo=ln2.
# 0.693359375 + (-2.12194440e-4) = 0.693147180.. good
ln2hi=f32(0.693359375); ln2lo=f32(-2.12194440e-4)
j,f=reduce_jf(xs, ln2hi, ln2lo)
# fit r = ds*2^-j vs f
r_meas=(ds.astype(f64)*np.exp2(-j)).astype(f64)
A=np.vander(f.astype(f64), 7, increasing=True)
coef,_,_,_=np.linalg.lstsq(A, r_meas, rcond=None)
c=[f32(v) for v in coef]
print('init coeffs:',[f'{v:.9g}' for v in c])
mdl=expf_model(xs,c,ln2hi,ln2lo)
print('init mismatches:',mism(mdl),'/',len(ds))

# --- coordinate descent on each f32 coeff (+ ln2lo) over +/- a few ULP ---
def f32_neighbors(v, span=6):
    b=np.float32(v).view(np.int32)
    return [np.int32(b+d).view(np.float32) for d in range(-span,span+1)]

best=list(c); bestm=mism(expf_model(xs,best,ln2hi,ln2lo))
params=list(range(7))
for it in range(6):
    improved=False
    for p in params:
        base=best[p]
        for cand in f32_neighbors(base, 8):
            trial=list(best); trial[p]=cand
            mm=mism(expf_model(xs,trial,ln2hi,ln2lo))
            if mm<bestm:
                bestm=mm; best=trial; improved=True
    # also nudge ln2lo
    for cand in f32_neighbors(ln2lo, 8):
        mm=mism(expf_model(xs,best,ln2hi,cand))
        if mm<bestm: bestm=mm; ln2lo=cand; improved=True
    print(f'iter {it}: mismatches={bestm}/{len(ds)}  ({100*bestm/len(ds):.2f}%)')
    if not improved or bestm==0: break

print('best coeffs:',[f'{np.float32(v).view(np.int32)}' for v in best],'(f32 bits)')
print('best coeffs val:',[f'{v:.9g}' for v in best])
print('ln2hi,ln2lo bits:',np.float32(ln2hi).view(np.int32),np.float32(ln2lo).view(np.int32))
print('final mismatches on search subset:',bestm,'/',len(ds))
