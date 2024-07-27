import numpy as np
import math
from typing import List

def t_len(t: np.ndarray) -> float: return sum([np.linalg.norm(t[i, :2] - t[i + 1, :2]) for i in range(len(t) - 1)])
def t2ps_its(T: np.ndarray, h: int):
    l = t_len(T)
    vec = np.array([T[0]] * h)
    vec[-1] = T[-1]
    if l < 1e-12: return vec
    e = l / (h - 1)
    a, b = 0, 0
    j = 0
    for i in range(1, len(T)):
        p1, p2 = T[i - 1], T[i]
        s = np.linalg.norm(p1[:2] - p2[:2], 2)
        if s<1e-12:continue
        b += s
        while b >= a:
            if j >= h: break
            vec[j] = (p2 - p1) * (1 - (b - a) / s) + p1
            j += 1
            a += e
    return vec
def itsps(TS:List[np.ndarray]):
    hT = int(np.mean([len(T) for T in TS]))
    ts=[t2ps_its(T,hT)[:,:2] for T in TS]
    return ts,hT

f_bk_def=lambda ct1,ct2:sum([np.linalg.norm((ct1[i] if ct1[i] is not None else 0) - (ct2[i] if ct2[i] is not None else 0)) for i in range(len(ct1))])/len(ct1) < 5e-3
def k_means(data, k, dis=None, center=None, max_epoch=300, f_mean=None, f_break=None, f_ep=None):
    center = np.random.choice(len(data),k,False) if center is None else center
    center = np.array([data[i] for i in center]) if isinstance(center[0],int) else center
    clas = np.zeros(len(data), dtype=int)
    dis = (lambda x, y: np.linalg.norm(x - y)) if dis is None else (dis if callable(dis) else lambda i, j: dis[i][j])
    f_mean=(lambda xs:np.mean(xs, axis=0)) if f_mean is None else f_mean
    f_break=f_bk_def if f_break is None else f_break
    for ep in range(max_epoch):
        if f_ep and callable(f_ep):f_ep()
        center_ = center.copy()
        for i in range(len(data)):
            d2c=np.zeros(k)
            for j in range(k): d2c[j]=np.inf if center[j] is None else dis(data[i],center[j])
            clas[i]=d2c.argmin()
        for j in range(k):
            if center[j] is None:continue
            xs=[data[i] for i in range(len(data)) if clas[i]==j]
            if len(xs)<=0:center[j]=None
            else:center[j] = f_mean(xs)
        if f_break(center, center_): break
    return clas
def KMST(data, n, k):
    ts, hT = itsps(data, n)
    sqh = np.sqrt(hT)
    dis = lambda x, y: np.linalg.norm((ts[x] if isinstance(x, int) else x) - (ts[y] if isinstance(y, int) else y)) / sqh
    return k_means(ts, k, dis)

class _Q3: 
    def __init__(Q, bbox, a: float, b: float, e: float):
        Q.rgx, Q.rgy, Q.e = a, b, e # a-x  b-y
        [Q.xa, Q.xb],[Q.ya, Q.yb]= bbox
        Q.ny, Q.nx = math.ceil(Q.yb / e) - math.floor(Q.ya / e), math.ceil(Q.xb / e) - math.floor(Q.xa / e)
        Q.w = np.zeros((Q.nx, Q.ny)) 
        Q.iys,Q.ixs=int(np.ceil(Q.rgy/e)),int(np.ceil(Q.rgx/e))
    def Update(Q,T:np.ndarray,w):
        xiA:np.ndarray= ((T[:,0]-Q.xa-Q.rgx/2) / Q.e+0.5).astype(int) 
        xiA[xiA<0]=0
        xiB:np.ndarray= ((T[:,0]-Q.xa+Q.rgx/2) / Q.e+0.5).astype(int) 
        xiB[xiB>Q.nx]=Q.nx
        yiA:np.ndarray= ((T[:,1]-Q.ya-Q.rgy/2) / Q.e+0.5).astype(int)
        yiA[yiA<0]=0
        yiB:np.ndarray= ((T[:,1]-Q.ya+Q.rgy/2) / Q.e+0.5).astype(int)
        yiB[yiB>Q.ny]=Q.ny
        ps=np.stack([np.stack([xiA+dx,yiA+dy]).T for dx in range(Q.ixs) for dy in range(Q.ixs)]) # shape:(dxy,len(T),2)
        mask=np.ones((len(ps),len(T)),dtype=bool)
        mask[ps[:,:,0]>=xiB]=False
        mask[ps[:,:,1]>=yiB]=False
        ps=ps[mask].reshape((-1,2))
        Q.w[ps[:,0],ps[:,1]]+=w
    def score(Q,T): 
        yi,xi=((T[:,1] - Q.ya) / Q.e+0.5).astype(int),((T[:,0] - Q.xa) / Q.e+0.5).astype(int)
        mask=np.ones(len(T),dtype=bool)
        mask[yi<0]=False
        mask[yi>=Q.ny]=False
        mask[xi<0]=False
        mask[xi>=Q.nx]=False
        n=int(sum(mask))
        return 0 if n==0 else sum(Q.w[xi[mask],yi[mask]])/n
def k_means_weight(data, k, weight, dis=None, center=None, max_epoch=300, f_break=None, f_ep=None):
    center = np.random.choice(len(data),k,False) if center is None else center
    center = np.array([data[i] for i in center]) if isinstance(center[0],int) else center
    clas = np.zeros(len(data), dtype=int)
    dis = (lambda x, y: np.linalg.norm(x - y)) if dis is None else (dis if callable(dis) else lambda i, j: dis[i][j])
    f_break=f_bk_def if f_break is None else f_break
    for ep in range(max_epoch):
        if f_ep and callable(f_ep):f_ep()
        center_ =center.copy()
        for i in range(len(data)):
            d2c=np.zeros(k)
            for j in range(k): d2c[j]=np.inf if center[j] is None else dis(data[i],center[j])
            clas[i]=d2c.argmin()
        for j in range(k):
            if center[j] is None:continue
            ws=[weight[i] for i in range(len(data)) if clas[i]==j]
            xs=[data[i]*weight[i] for i in range(len(data)) if clas[i]==j]
            if len(xs)<=0:center[j]=None
            else:center[j] = np.sum(xs, axis=0)/sum(ws)
        if f_break(center, center_): break
    return clas
bbox_beijing = [[39.8, 40.05], [116.25, 116.5]]
def QKMST(data,n,k,ab, e):
    ts, hT = itsps(data, n)
    sqh = np.sqrt(hT)
    dis = lambda x, y: np.linalg.norm((ts[x] if isinstance(x, int) else x) - (ts[y] if isinstance(y, int) else y)) / sqh
    q=_Q3(bbox_beijing,ab,ab,e)
    [q.Update(T,1) for T in ts]
    dense=np.array([q.score(t) for t in ts])
    return k_means_weight(ts, k,dense,dis)
