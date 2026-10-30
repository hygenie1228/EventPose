"""Target-mask RGB crops with calibrated intrinsics and identity acceptance gates."""
import cv2
import math
import numpy as np
from PIL import Image,ImageOps

PROTOCOL='target_foreground_head_v1'
STRICT_PROTOCOLS={'native_rgb':'native_rgb_strict_mask_target_head_v2',
                  'native_rgb_static_crop':'native_rgb_static_crop_strict_mask_target_head_v2'}
BODY_PROTOCOLS={'native_rgb':'native_rgb_strict_mask_body_box_v3',
                'native_rgb_static_crop':'native_rgb_static_crop_strict_mask_body_box_v3'}


def target_body_box_iou(joints,K,support):
    """Compare projected body extent with the target mask, using no GT pose."""
    points=np.asarray(joints[:22],np.float64)@np.asarray(K,np.float64).T
    if not np.isfinite(points).all() or (points[:,2]<=.05).any():return 0.
    xy=points[:,:2]/points[:,2:];height,width=support.shape
    lo=np.maximum(xy.min(0),[0,0]);hi=np.minimum(xy.max(0),[width,height])
    ys,xs=np.where(support)
    if not len(xs) or (hi<=lo).any():return 0.
    target_lo=np.array([xs.min(),ys.min()]);target_hi=np.array([xs.max()+1,ys.max()+1])
    intersection=np.maximum(np.minimum(hi,target_hi)-np.maximum(lo,target_lo),0).prod()
    union=(hi-lo).prod()+(target_hi-target_lo).prod()-intersection
    return float(intersection/max(union,1e-8))


def fixed_target_box(mask_union,image_shape,K,margin=.1,reference_fov=60):
    """One clip-wide RGB viewport from mask bounds; no pose/GT parameters."""
    height,width=map(int,image_shape);left,top,right,bottom=map(int,mask_union)
    assert 0<=left<right<=width and 0<=top<bottom<=height
    desired=2*float(K[0][0])*math.tan(math.radians(reference_fov/2))
    cw=min(width,math.ceil(max(desired,(right-left)*(1+margin))))
    ch=min(height,math.ceil(max(desired,(bottom-top)*(1+margin))))
    x=max(0,min(width-cw,math.floor((left+right-cw)/2)))
    y=max(0,min(height-ch,math.floor((top+bottom-ch)/2)))
    assert x<=left and y<=top and x+cw>=right and y+ch>=bottom
    return [x,y,x+cw,y+ch]


def prepare_native_rgb_box(rgb,K,box,size=896):
    """A fixed viewport change; retain all RGB pixels and transform K exactly."""
    left,top,right,bottom=map(int,box);height,width=rgb.shape[:2]
    assert 0<=left<right<=width and 0<=top<bottom<=height
    cw,ch=right-left,bottom-top
    image=ImageOps.contain(Image.fromarray(rgb[top:bottom,left:right]),(size,size),method=Image.Resampling.LANCZOS)
    nw,nh=image.size;px,py=(size-nw)//2,(size-nh)//2
    image=ImageOps.expand(image,border=(px,py,size-nw-px,size-nh-py))
    camera=np.asarray(K,np.float32).copy();camera[0,2]-=left;camera[1,2]-=top
    camera[0]*=nw/cw;camera[1]*=nh/ch;camera[0,2]+=px;camera[1,2]+=py
    return image,camera,dict(box=[left,top,right,bottom],scale=[nw/cw,nh/ch],padding=[px,py],mask_valid=True)


def prepare_target_crop(rgb,mask,K,margin=.2,size=896,zoom=False,strict_target_mask=False):
    height,width=rgb.shape[:2]
    gray=cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY) if mask.ndim==3 else mask
    if gray.shape!=(height,width):gray=cv2.resize(gray,(width,height),interpolation=cv2.INTER_NEAREST)
    binary=gray>127;yy,xx=np.where(binary)
    if len(xx):
        left,right=int(xx.min()),int(xx.max())+1;top,bottom=int(yy.min()),int(yy.max())+1
        dx=max(8,int(round((right-left)*margin)));dy=max(8,int(round((bottom-top)*margin)))
        left=max(0,left-dx);right=min(width,right+dx);top=max(0,top-dy);bottom=min(height,bottom+dy)
    else:left,top,right,bottom=0,0,width,height
    target_box=[left,top,right,bottom]
    # Keep the original model's field of view and camera constant. The mask
    # determines identity; changing zoom every frame perturbs depth estimates.
    if not zoom:left,top,right,bottom=0,0,width,height
    foreground=rgb.copy();foreground[~binary]=np.array([124,116,104],np.uint8)
    image=Image.fromarray(foreground[top:bottom,left:right])
    crop_width,crop_height=image.size
    image=ImageOps.contain(image,(size,size),method=Image.Resampling.LANCZOS)
    nw,nh=image.size;pad_x,pad_y=(size-nw)//2,(size-nh)//2
    image=ImageOps.expand(image,border=(pad_x,pad_y,size-nw-pad_x,size-nh-pad_y))
    camera=np.asarray(K,np.float32).copy();camera[0,2]-=left;camera[1,2]-=top
    camera[0]*=nw/crop_width;camera[1]*=nh/crop_height;camera[0,2]+=pad_x;camera[1,2]+=pad_y
    info=dict(box=[left,top,right,bottom],target_box=target_box,zoom=zoom,scale=[nw/crop_width,nh/crop_height],padding=[pad_x,pad_y],mask_valid=bool(len(xx)))
    support=binary.copy() if strict_target_mask else cv2.dilate(binary.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15)))>0
    return image,camera,support,info


def target_acceptance(joints,K,support):
    """Accept only poses whose projected torso belongs to the supplied target."""
    projected=np.asarray(joints[:22])@np.asarray(K).T
    uv=projected[:,:2]/np.maximum(projected[:,2:],1e-8)
    height,width=support.shape
    valid=np.isfinite(uv).all(1)&(projected[:,2]>.05)&(uv[:,0]>=0)&(uv[:,0]<width)&(uv[:,1]>=0)&(uv[:,1]<height)
    inside=np.zeros(22,bool);xy=np.rint(uv[valid]).astype(int)
    xy[:,0]=np.clip(xy[:,0],0,width-1);xy[:,1]=np.clip(xy[:,1],0,height-1)
    inside[valid]=support[xy[:,1],xy[:,0]]
    torso=inside[[0,3,6,9,12,15]].mean();overlap=inside.mean()
    return bool(torso>=.5 and overlap>=.35),float(overlap),float(torso)


def target_head_acceptance(location,info,support):
    """The detector's head query must belong to the provided target mask."""
    xy=(np.asarray(location,np.float64)-np.asarray(info['padding']))/np.asarray(info['scale'])+np.asarray(info['box'][:2])
    if not np.isfinite(xy).all():return False,xy
    x,y=np.rint(xy).astype(int);height,width=support.shape
    return bool(0<=x<width and 0<=y<height and support[y,x]),xy


def select_target_head(people,native_K,image_K,image_shape,mask,strict_target_mask=False,body_box_min_iou=None,joint_regressor=None):
    """Select by supplied 2D mask and confidence; no GT pose enters selection."""
    height,width=map(int,image_shape)
    gray=cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY) if mask.ndim==3 else mask
    if gray.shape!=(height,width):gray=cv2.resize(gray,(width,height),interpolation=cv2.INTER_NEAREST)
    support=gray>127 if strict_target_mask else cv2.dilate((gray>127).astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15)))>0
    sx,sy=image_K[0,0]/native_K[0,0],image_K[1,1]/native_K[1,1]
    info=dict(box=[0,0,width,height],scale=[sx,sy],
              padding=[image_K[0,2]-sx*native_K[0,2],image_K[1,2]-sy*native_K[1,2]])
    accepted=[p for p in people if target_head_acceptance(p['loc'].detach().float().cpu().numpy(),info,support)[0]]
    if body_box_min_iou is not None:
        assert strict_target_mask and joint_regressor is not None
        ranked=[]
        for person in accepted:
            joints=(joint_regressor@person['v3d'].float()).detach().cpu().numpy()
            overlap=target_body_box_iou(joints,native_K,support)
            if overlap>=body_box_min_iou:ranked.append((overlap,float(person['scores']),person))
        return max(ranked,key=lambda row:row[:2])[2] if ranked else None
    return max(accepted,key=lambda p:float(p['scores'])) if accepted else None
