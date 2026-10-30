"""Hand conventions shared by preparation, learning and evaluation (metres)."""
import numpy as np


def joint_indices(side):
    from smplx.joint_names import JOINT_NAMES
    names=[side+'_wrist']
    for finger in ('thumb','index','middle','ring','pinky'):
        names += [f'{side}_{finger}{i}' for i in (1,2,3)]+[side+'_'+finger]
    return [JOINT_NAMES.index(n) for n in names]


def crop_transform(box, right, size=256, rescale=2.5):
    """Official HaMeR square crop with 192:256 bbox expansion and left flip."""
    box=np.asarray(box,np.float64);center=(box[:2]+box[2:])/2
    width,height=(box[2:]-box[:2])*rescale
    edge=max(width*256/192,height)
    if not np.isfinite(edge) or edge<=1:raise ValueError('Invalid hand box')
    scale=size/edge
    # Matches upstream cv2 affine: centre maps to size/2, left flip uses W-1.
    A=np.array([[scale,0,size/2-scale*center[0]],
                [0,scale,size/2-scale*center[1]],[0,0,1.]])
    if not right:A=np.array([[-1,0,size],[0,1,0],[0,0,1.]])@A
    return A.astype(np.float32),float(edge)


def projected_box(vertices,K,margin=.15):
    vertices=np.asarray(vertices)
    if not np.isfinite(vertices).all() or (vertices[:,2]<=.05).any():return None
    uv=vertices@K.T;uv=uv[:,:2]/uv[:,2:]
    low,high=uv.min(0),uv.max(0);pad=(high-low).max()*margin
    return np.r_[low-pad,high+pad].astype(np.float32)


def box_overlap(box,shape):
    if box is None:return 0.
    low=np.maximum(box[:2],0);high=np.minimum(box[2:],np.asarray(shape)[::-1])
    return float(np.maximum(high-low,0).prod()/np.maximum(box[2:]-box[:2],1).prod())


def apply_crop(array,A,size=256):
    import cv2
    # CHW works for both 2-polarity/6-bin tensors and RGB.
    value=cv2.warpAffine(array.transpose(1,2,0),A[:2],(size,size),flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    if value.ndim==2:value=value[:,:,None]
    return value.transpose(2,0,1)


def rectify_boxes(boxes,source_K,source_distortion,rectified_K):
    """Map source-resolution GT box boundaries into the processed RGB pixels."""
    import cv2
    boxes=np.asarray(boxes);shape=boxes.shape;flat=boxes.reshape(-1,4);result=np.empty_like(flat,dtype=np.float32)
    alpha=np.linspace(0,1,9,dtype=np.float32)
    for i,box in enumerate(flat):
        x1,y1,x2,y2=box
        x=x1+(x2-x1)*alpha;y=y1+(y2-y1)*alpha
        edge=np.r_[np.c_[x,np.full_like(x,y1)],np.c_[x,np.full_like(x,y2)],
                   np.c_[np.full_like(y,x1),y],np.c_[np.full_like(y,x2),y]]
        uv=cv2.undistortPoints(edge[:,None].astype(np.float64),source_K,source_distortion,P=rectified_K)[:,0]
        result[i]=np.r_[uv.min(0),uv.max(0)]
    return result.reshape(shape)
