"""Causal native-resolution display matching the documented event comparison."""
from pathlib import Path
import zipfile
import numpy as np
import cv2

def _slice(path,key,start,end):
    with zipfile.ZipFile(path) as archive,archive.open(key+'.npy') as stream:
        version=np.lib.format.read_magic(stream);shape,fortran,dtype=np.lib.format._read_array_header(stream,version)
        assert not fortran and len(shape)==1 and 0<=start<=end<=shape[0]
        stream.seek(start*dtype.itemsize,1)
        return np.frombuffer(stream.read((end-start)*dtype.itemsize),dtype=dtype).copy()

def load_window(path,start_ns,end_ns):
    with np.load(Path(path),allow_pickle=False) as z:
        times=z['timestamp_ns'].astype(np.int64);ptr=z['indptr'];shape=tuple(map(int,z['image_shape']))
    lo,hi=np.searchsorted(times,[int(start_ns),int(end_ns)],side='right');a,b=map(int,ptr[[lo,hi]])
    return dict(times=times[lo:hi],ptr=ptr[lo:hi+1].astype(np.int64)-a,indices=_slice(path,'indices',a,b),
                positive=_slice(path,'data',a,b)>0,shape=shape)

def intensity(count):return np.sqrt(np.minimum(np.log1p(count)/np.log(9),1)).astype(np.float32)

def counts(window,start_ns,end_ns):
    lo,hi=np.searchsorted(window['times'],[int(start_ns),int(end_ns)],side='right');a,b=map(int,window['ptr'][[lo,hi]])
    ix=window['indices'][a:b];pos=window['positive'][a:b];h,w=window['shape']
    return (np.bincount(ix[pos],minlength=h*w).reshape(h,w),np.bincount(ix[~pos],minlength=h*w).reshape(h,w))

def render(window,end_ns,encoding='count33',supported=True):
    end_ns=int(end_ns);h,w=window['shape']
    if encoding=='count33':
        p,n=counts(window,end_ns-33_000_000,end_ns)
        if supported:
            active=(p+n)>0;support=cv2.boxFilter(active.astype(np.float32),-1,(3,3),normalize=False,borderType=cv2.BORDER_CONSTANT)
            keep=active&(support>=2);p=p*keep;n=n*keep
        p,n=intensity(p),intensity(n)
        return np.uint8(np.clip(np.stack([p,.65*n+.25*p,n],-1),0,1)*255)
    if encoding=='bins100':
        edges=end_ns-100_000_000+np.arange(4,dtype=np.int64)*100_000_000//3;channels=[]
        for i in range(3):
            p,n=counts(window,int(edges[i]),int(edges[i+1]));channels.append(intensity(p+n))
        return np.uint8(np.stack(channels,-1)*255)
    assert encoding=='surface20'
    lo,hi=np.searchsorted(window['times'],[end_ns-200_000_000,end_ns],side='right');a,b=map(int,window['ptr'][[lo,hi]])
    ix=window['indices'][a:b];pos=window['positive'][a:b]
    age=np.repeat((window['times'][lo:hi]-np.int64(end_ns)).astype(np.float64)/1e9,np.diff(window['ptr'][lo:hi+1]))
    assert not len(age) or np.all(age<=0),'Future observation or unsigned timestamp underflow'
    lp=np.full(h*w,-np.inf);ln=lp.copy();np.maximum.at(lp,ix[pos],age[pos]);np.maximum.at(ln,ix[~pos],age[~pos])
    p=np.exp(lp/.02).reshape(h,w);n=np.exp(ln/.02).reshape(h,w)
    return np.uint8(np.clip(np.stack([p,.65*n+.25*p,n],-1),0,1)*255)
