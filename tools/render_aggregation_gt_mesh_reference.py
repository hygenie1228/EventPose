"""Render actual dataset SMPL-X GT as a clearly labelled output-format reference."""
from pathlib import Path
import json
import numpy as np
import h5py,torch,smplx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'exp/261029/dynamic_aggregation_paper_figure'
meta=json.loads((OUT/'A_source.json').read_text());src=Path(meta['source']);end=meta['timestamp_ns'];torch.set_num_threads(2)
with h5py.File(src/'label.h5','r',locking=False) as f:
 times=f['frames/source_rgb_timestamp_ns'][:];row=max(0,int(np.searchsorted(times,end,side='right'))-1)
 pose=f['params/smplx_pose_axis_angle'][row,0];betas=f['params/smplx_betas'][row,0];tr=f['params/translation_world_m'][row,0]
model=smplx.create(str(ROOT/'data/human_model_files'),model_type='smplx',gender='neutral',num_betas=16,use_pca=False)
p=torch.from_numpy(pose[None]);b=torch.from_numpy(betas[None]);translation=torch.from_numpy(tr[None])
with torch.no_grad():
 out=model(betas=b,global_orient=p[:,:3],body_pose=p[:,3:66],jaw_pose=p[:,66:69],leye_pose=p[:,69:72],reye_pose=p[:,72:75],left_hand_pose=p[:,75:120],right_hand_pose=p[:,120:165],transl=translation,expression=torch.zeros((1,10)))
 verts=out.vertices[0].numpy();joints=out.joints[0,:22].numpy()
with np.load(src/'event.npz') as z:R=z['rgb_calibration__rgb__rotation_matrix'];t=z['rgb_calibration__rgb__translation']
verts=verts@R.T+t;joints=joints@R.T+t
# Cartesian display: horizontal X, depth Z, vertical -Y.
v=verts[:,[0,2,1]].copy();v[:,2]*=-1;j=joints[:,[0,2,1]].copy();j[:,2]*=-1
center=(v.min(0)+v.max(0))/2;v-=center;j-=center;faces=model.faces;tri=v[faces]
n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-9)
light=np.array([-.4,-1.,1.]);light/=np.linalg.norm(light);shade=.55+.35*np.maximum(n@light,0)
colors=shade[:,None]*np.array([.83,.86,.9])[None]
fig=plt.figure(figsize=(4.2,7.8),facecolor='white');ax=fig.add_axes([0,0,1,1],projection='3d');ax.computed_zorder=False
ax.add_collection3d(Poly3DCollection(tri,facecolors=colors,edgecolors='none',linewidths=0,zsort='average',zorder=1))
# Scatter points are actual GT joints; coloured by body side for readability.
cs=['#387bc1']*22
for i in [1,4,7,10,13,16,18,20]:cs[i]='#f09c2d'
for i in [2,5,8,11,14,17,19,21]:cs[i]='#8a63c4'
ax.scatter(j[:,0],j[:,1]-.035,j[:,2],c=cs,s=15,depthshade=False,edgecolors='white',linewidths=.35,zorder=10)
for child in range(1,22):
 parent=int(model.parents[child]);ax.plot(j[[parent,child],0],j[[parent,child],1]-.035,j[[parent,child],2],c='#627085',lw=.65,zorder=9)
rx=max(np.ptp(v[:,0])/2,.32);rz=np.ptp(v[:,2])/2
ax.set_xlim(-rx*1.15,rx*1.15);ax.set_ylim(-.4,.4);ax.set_zlim(-rz*1.08,rz*1.08)
ax.set_box_aspect((rx*2.3,.8,rz*2.16));ax.view_init(elev=5,azim=-82);ax.set_proj_type('ortho');ax.set_axis_off()
fig.savefig(OUT/'gt_mesh_reference.png',dpi=230,transparent=True,pad_inches=0);plt.close(fig)
from PIL import Image
with Image.open(OUT/'gt_mesh_reference.png') as im:
 alpha=np.asarray(im.getchannel('A'));yy,xx=np.nonzero(alpha>5);bounds=(max(0,int(xx.min())-12),max(0,int(yy.min())-12),min(im.width,int(xx.max())+13),min(im.height,int(yy.max())+13));trim=im.crop(bounds);trim.save(OUT/'gt_mesh_reference.png')
(OUT/'gt_mesh_reference.json').write_text(json.dumps(dict(session=meta['session'],label_row=row,label_timestamp_ns=int(times[row]),query_timestamp_ns=end,render='Actual SMPL-X GT parameters and official body model; neutral model; camera transformed',network_prediction=False,output_reference_only=True),indent=2))
print('GT reference rendered')
