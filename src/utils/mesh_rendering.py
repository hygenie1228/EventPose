"""Pure NumPy/OpenCV mesh rasterization; no dataset access."""
import cv2
import numpy as np


def render(image,vertices,faces,K,alpha=.65):
    out=image.copy()
    for person,v in enumerate(vertices):
        triangles=v[faces]
        normal=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
        normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-9)
        brightness=.35+.65*np.abs(normal@np.array([.3,-.4,-.866]))
        base=np.array([(245,170,65),(90,145,250)][person%2])
        colors=np.clip(brightness[:,None]*base,0,255).astype(np.uint8)
        projected=v@K.T
        uv=projected[:,:2]/np.maximum(projected[:,2:],1e-5)
        polygons=np.clip(uv[faces],-10000,10000).astype(np.int32)
        for idx in np.argsort(triangles[:,:,2].mean(axis=1))[::-1]:
            if triangles[idx,:,2].min()>.01:
                cv2.fillConvexPoly(out,polygons[idx],tuple(map(int,colors[idx])),lineType=cv2.LINE_AA)
    return cv2.addWeighted(out,alpha,image,1-alpha,0)


def panel(image,title):
    height,width=image.shape[:2]
    scale=min(640/width,330/height)
    resized=cv2.resize(image,(round(width*scale),round(height*scale)),interpolation=cv2.INTER_AREA)
    out=np.full((360,640,3),25,np.uint8)
    x=(640-resized.shape[1])//2;y=30+(330-resized.shape[0])//2
    out[y:y+resized.shape[0],x:x+resized.shape[1]]=resized
    cv2.putText(out,title,(10,22),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1,cv2.LINE_AA)
    return out
