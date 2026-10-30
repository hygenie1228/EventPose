"""GT-free original RGB inference batch consistency; never selects a model."""
from pathlib import Path
import json,sys,time
import cv2,numpy as np,torch
import torchvision.transforms as T
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from multihmr.track2_data import OUTPUT,SOURCE,save_json
from multihmr.target_crop import prepare_native_rgb_box,select_target_head
from utils.pose_evaluation import configure_torch,load_multihmr


def main():
    configure_torch();cv2.setNumThreads(1)
    name='blue_easter_basket-place-10';manifest=json.loads((OUTPUT/'manifest.json').read_text());record=manifest['samples'][name]
    frames=[0,record['frames']//2]+list(range(record['frames']-4,record['frames']))
    cap=cv2.VideoCapture(str(SOURCE/name/'video.mp4'));maskcap=cv2.VideoCapture(str(SOURCE/name/'processed/human_mask.mp4'))
    K=np.asarray(record['K'],np.float32);items=[];norm=T.Compose([T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
    for frame in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES,frame);maskcap.set(cv2.CAP_PROP_POS_FRAMES,frame)
        ok,bgr=cap.read();ok2,mask=maskcap.read();assert ok and ok2
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);image,camera,_=prepare_native_rgb_box(rgb,K,[0,0,record['width'],record['height']])
        items.append(dict(frame=frame,image=norm(image),camera=camera,mask=mask))
    cap.release();maskcap.release();model=load_multihmr();ids=[];old=model.detection
    def detection(*args,**kwargs):
        value=old(*args,**kwargs);ids[:]=value[2][0].detach().cpu().tolist();return value
    model.detection=detection;reg=model.smpl_layer[f'neutral_{model.num_betas}'].bm_x.J_regressor
    def infer(chosen):
        images=torch.stack([x['image'] for x in chosen]).cuda();cameras=torch.tensor(np.stack([x['camera'] for x in chosen])).cuda()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):people=model(images,K=cameras,is_training=False,det_thresh=.3,nms_kernel_size=3)
        groups=[[] for _ in chosen]
        for i,p in zip(ids,people):groups[i].append(p)
        result={}
        for i,item in enumerate(chosen):
            p=select_target_head(groups[i],K,item['camera'],[record['height'],record['width']],item['mask']);assert p is not None
            joints=(reg@p['v3d'].float()).detach().cpu().numpy()
            result[item['frame']]=dict(joints=joints,root_Z_m=float(joints[0,2]),score=float(p['scores']),
                loc=p['loc'].detach().float().cpu().numpy().tolist(),all_detected_people=len(groups[i]))
        return result
    single={}
    for item in items:single.update(infer([item]))
    batches={'tail4':infer(items[-4:]),'mixed6':infer(items)};comparisons={};largest=0
    for label,values in batches.items():
        comparisons[label]={}
        for frame,value in values.items():
            difference=float(np.max(np.linalg.norm(value['joints']-single[frame]['joints'],axis=1))*100);largest=max(largest,difference)
            comparisons[label][str(frame)]=dict(max_joint_difference_cm=difference,
                single={k:v for k,v in single[frame].items() if k!='joints'},batch={k:v for k,v in value.items() if k!='joints'})
    save_json(OUTPUT/'native_batch_consistency.json',dict(sample=name,frames=frames,comparisons=comparisons,
        max_joint_difference_cm=largest,catastrophic_batch_dependency_over10cm=largest>10,
        GT3D_read=False,GT_used_for_selection=False,not_used_for_model_or_recipe_selection=True,
        original_weights=True,original_RGB=True,head_gate_unchanged=True,checked_unix=time.time()))
    print('batch_consistency_complete max_joint_difference_cm',largest,flush=True)


if __name__=='__main__':main()
