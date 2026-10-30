"""Compact Figure-2-style overview with an actual Train initialization example."""
import hashlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from goal_runtime_control import register_job


def main():
    goal=json.loads((ROOT/'_agents/research/261027/device34_fusion_repair_goal_15h.json').read_text())
    register_job(goal,os.getpid(),'fusion_repair_paper_figure_CPU')
    os.environ.update(EVENTPOSE_MULTISENSOR_OUT=goal['output_root'],XFORMERS_DISABLED='1',CUDA_VISIBLE_DEVICES='')
    import numpy as np
    import h5py
    out=Path(goal['output_root'])/'figures';assets=out/'paper_style_assets';assets.mkdir(exist_ok=True)
    asset=assets/'actual_initialization_example.npz'
    if not asset.exists():
        import torch
        from torch.utils.data import DataLoader
        from utils.pose_evaluation import load_multihmr
        from multihmr.multisensor_repair import build_repair_model,load_verified_initialization,REPAIR_LAYOUTS
        from multihmr.multisensor_data import Frames,collate,input_keys
        torch.set_num_threads(2)
        config=json.loads((Path(goal['output_root'])/'configs/rgb3_event4.json').read_text())
        torch.manual_seed(config['seed']);base=load_multihmr(device='cpu')
        model=build_repair_model(base,config).eval();provenance=load_verified_initialization(model,config)
        data=Frames('training',condition='rgb3_event4',sensor_layout=REPAIR_LAYOUTS['rgb3_event4'],limit=1)
        batch=next(iter(DataLoader(data,batch_size=1,collate_fn=collate,num_workers=0)))
        capture={};original=base.detection
        def detection(z,*args,**kwargs):
            capture['scores']=torch.sigmoid(base.mlp_classif(z)).detach().numpy()[0,:,0].reshape(64,64)
            return original(z,*args,**kwargs)
        base.detection=detection
        with torch.inference_mode():people=model(**input_keys(batch),is_training=False,det_thresh=.3,nms_kernel_size=3)
        base.detection=original
        assert people
        row=data.records[0]
        with h5py.File(Path(goal['output_root'])/'cache'/(row['session']+'.h5'),'r',locking=False) as f:
            rgb=f['rgb3'][row['cache_row']].transpose(1,2,0)
            event=f['event4'][row['cache_row']]
        mesh=np.stack([p['v3d'].numpy() for p in people]);loc=np.stack([p['loc'].numpy() for p in people])
        faces=next(iter(base.smpl_layer.values())).bm_x.faces
        np.savez_compressed(asset,rgb=rgb,event=event,scores=capture['scores'],vertices=mesh,
                            faces=faces,K=batch['K'][0].numpy(),locations=loc)
        (assets/'example_provenance.json').write_text(json.dumps(dict(
            condition='rgb3_event4',source_role='training',row=row,initialization=provenance,
            checkpoint_stage='Verified initialization before the new fine-tuning',
            GT_used_as_inference_input=False,Test_used=False,device='CPU',people=len(people)),indent=2)+'\n')
        for h in data.handles.values():h.close()
        del model,base
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Circle,Rectangle
    from matplotlib.collections import PolyCollection
    plt.rcParams.update({'font.family':'DejaVu Serif','pdf.fonttype':42,'svg.fonttype':'none'})
    z=np.load(asset);rgb=z['rgb'];event=z['event'];scores=z['scores'];K=z['K']
    fig,ax=plt.subplots(figsize=(24,7.1))
    fig.subplots_adjust(left=.01,right=.99,top=.98,bottom=.035)
    ax.set(xlim=(0,102),ylim=(0,30));ax.axis('off')
    blue='#7ba6d1';orange='#efba79';edge='#263645';gray='#64748b';colors=['#edc0a6','#8aad5d','#f1be36']
    def text(x,y,t,size=13,**kw):ax.text(x,y,t,ha='center',va='center',fontsize=size,**kw)
    def box(x,y,w,h,label,color=blue,size=14):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.12,rounding_size=.65',fc=color,ec=edge,lw=1.2))
        text(x+w/2,y+h/2,label,size)
    def arrow(a,b,dashed=False):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,lw=1.2,
                     color=edge,linestyle='--' if dashed else '-'))
    def path(xs,ys,dashed=False):ax.plot(xs,ys,color=edge,lw=1.2,ls='--' if dashed else '-')
    def circ(x,y,symbol,r=.85,color='white'):
        ax.add_patch(Circle((x,y),r,ec=edge,fc=color,lw=1.2));text(x,y,symbol,15)

    # Main path: image → ViT → fused detections → human queries → HPH → SMPL-X.
    ax.imshow(rgb,extent=(1,12,16,27),zorder=2)
    text(6.5,28.2,'Primary input',14)
    text(6.5,14.6,'RGB / adapted event',11,color=gray)
    box(15,18.9,7,5.2,'ViT-L/14')
    arrow((12,21.5),(15,21.5));arrow((22,21.5),(25.1,21.5))
    text(18.5,17.1,'896 × 896',11,color=gray)
    circ(26,21.5,'+',r=.9,color=orange)
    text(26,25.8,'Fusion',13)
    arrow((26.9,21.5),(30,21.5))

    # The thumbnail is a genuine prediction map, with a schematic coarse grid.
    ax.imshow(rgb,extent=(30,41,16,27),alpha=.66,zorder=2)
    heat=np.ma.masked_less(scores,.05)
    ax.imshow(heat,extent=(30,41,16,27),cmap='magma',alpha=.68,vmin=0,vmax=1,zorder=3)
    for v in np.linspace(30,41,9):ax.plot([v,v],[16,27],color='#45566d',lw=.65,zorder=4)
    for v in np.linspace(16,27,9):ax.plot([30,41],[v,v],color='#45566d',lw=.65,zorder=4)
    text(35.5,28.2,'Detection + offset',14)
    text(35.5,14.6,'64 × 64 fused tokens',11,color=gray)
    arrow((41,21.5),(42.4,21.5));circ(43.2,21.5,'C',r=.75)
    text(43.2,24.1,'concat.',10,color=gray)
    arrow((44,21.5),(46.2,21.5))
    for i,color in enumerate(colors):
        ax.add_patch(Rectangle((46.5,17.2+i*3.1),2.3,2.3,fc=color,ec=edge,lw=1.1))
    text(47.7,28.2,'Human queries',14)
    arrow((48.8,21.5),(52.5,21.5))
    box(52.5,17.1,11.2,9,'Human\nPerception\nHead',size=17)
    text(58.1,15.5,'HPH',11,color=gray)
    arrow((63.8,21.5),(67,21.5))
    ax.add_patch(FancyBboxPatch((67.2,17.3),8,8.4,boxstyle='round,pad=.15,rounding_size=.6',fc='#f4f6f8',ec=gray,lw=1))
    for y,label in [(24.1,'Pose'),(21.5,'Shape'),(18.9,'Transl.')]:
        ax.add_patch(FancyBboxPatch((67.8,y-.9),6.8,1.8,boxstyle='round,pad=.1,rounding_size=.3',fc='white',ec=gray,lw=.9))
        text(71.2,y,label,12)
    text(71.2,28.2,'Human predictions',14)
    arrow((75.5,21.5),(78,21.5));box(78,19,7,5,'SMPL-X',size=14)
    arrow((85.2,21.5),(88,21.5))

    # Real initialization mesh, rasterized only inside the otherwise vector figure.
    ax.imshow(rgb,extent=(88,99,16,27),zorder=2)
    for i,verts in enumerate(z['vertices']):
        hom=verts@K.T;uv=hom[:,:2]/hom[:,2:]
        faces=z['faces'];tri=verts[faces]
        normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-8)
        shade=.52+.48*np.abs(normal@np.array([.2,-.3,.93]))
        base_color=np.array([.64,.43,.75]);face_color=np.clip(shade[:,None]*base_color[None],0,1)
        polygon=uv[faces];polygon[:,:,0]=88+polygon[:,:,0]/896*11;polygon[:,:,1]=27-polygon[:,:,1]/896*11
        order=np.argsort(tri[:,:,2].mean(1))[::-1]
        collection=PolyCollection(polygon[order],facecolors=face_color[order],edgecolors='none',rasterized=True,zorder=5)
        ax.add_collection(collection)
    text(93.5,28.2,'3D mesh',14)

    # One compact new branch, instead of a second row of explanatory boxes.
    ev=np.ones((224,224,3),np.float32)*.97
    pos=event[0].astype(np.float32)/255;neg=event[1].astype(np.float32)/255
    ev[:,:,0]-=.65*neg;ev[:,:,1]-=.8*np.maximum(pos,neg);ev[:,:,2]-=.65*pos
    ax.imshow(np.clip(ev,0,1),extent=(1,10,2.5,11.5),zorder=2)
    text(5.5,12.8,'Auxiliary event',13)
    text(5.5,1,'2 × 224 × 224',10,color=gray)
    arrow((10,7),(14,7));box(14,4,7,6,'Event\nCNN',orange,size=14)
    arrow((21.2,7),(23,7));box(23,4,7.8,6,'Ray\nattention',orange,size=13)
    path([23.2,23.2,24.4],[21.5,12.3,12.3],dashed=True)
    arrow((24.4,12.3),(24.4,10.2),dashed=True)
    text(21.3,13.5,'query',10,color=gray)
    arrow((26,10.2),(26,20.6))
    text(28.1,13.6,'ΔF',12,color='#94602e')
    text(26.9,2.5,'Kp, Ka, R, t',11,color=gray)
    arrow((26.9,3),(26.9,4),dashed=True)
    text(17.5,2.5,'128 channels',10,color=gray)

    # Intrinsic ray encoding enters the same tokens as in the original model.
    text(34.5,7,'Kp',12)
    arrow((36,7),(38.5,7),dashed=True)
    box(38.5,4.5,8,5,'Fourier',size=13)
    path([43.2,43.2],[9.7,20.7],dashed=True)
    # All tokens become K/V; detected tokens become the human queries.
    path([44.1,45.1,45.1],[20.7,19.5,11.8])
    arrow((45.1,11.8),(49.2,11.8))
    for i in range(8):
        color=['#dae6f3','#dae6f3',colors[0],'#dae6f3',colors[1],'#dae6f3',colors[2],'#dae6f3'][i]
        ax.add_patch(Rectangle((49.5+i*3,10.5),2.5,2.5,fc=color,ec=gray,lw=.9))
    text(75,11.8,'…',18)
    text(61.4,8.7,'Keys / values: all fused tokens',12)
    arrow((58.1,13.1),(58.1,17))
    text(80,6.6,'Blue: Multi-HMR    Orange: added event fusion',11,color=gray)
    text(80,3.9,'Available-gated residual; zero-initialized ΔF',11,color=gray)
    ax.text(1,-.2,'Example images / mesh: fixed Train frame at verified initialization; not a new Test result.',
            fontsize=9,color=gray,va='bottom')
    paths=[]
    for ext in ['png','svg','pdf']:
        dest=out/('신체센서융합_논문형_전체구조.'+ext);fig.savefig(dest,dpi=200,facecolor='white');paths.append(str(dest))
    plt.close(fig)
    record=dict(created_kst=datetime.now(timezone(timedelta(hours=9))).isoformat(),
                layout_reference='https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03364.pdf#page=3',
                description='New illustration in the horizontal Figure 2 layout; not a copied paper figure',
                actual_example=str(asset),example_sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),
                outputs={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
                source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'논문형_전체구조_출처.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(status='rendered',outputs=paths),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
