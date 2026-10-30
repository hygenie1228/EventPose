"""Full frozen Test evaluation for GT-cropped HaMeR hand reconstruction."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from hamer.hand_data import HandFrames,to_device,model_inputs
from hamer.event_model import load_model,file_sha256
from utils.hand_metrics import score_session


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2));tmp.replace(path)


def preview(dataset,predictions,dest):
    from PIL import Image,ImageDraw
    ids=np.linspace(0,len(dataset)-1,min(8,len(dataset)),dtype=int)
    canvas=Image.new('RGB',(256*4,280*2),'white');draw=ImageDraw.Draw(canvas)
    for slot,i in enumerate(ids):
        sample=dataset[int(i)];rgb=sample['image'].numpy()*np.array([.229,.224,.225])[:,None,None]+np.array([.485,.456,.406])[:,None,None]
        im=Image.fromarray(np.rint(rgb.transpose(1,2,0)*255).clip(0,255).astype(np.uint8));pen=ImageDraw.Draw(im)
        for joints,color in [(sample['joints'].numpy(),'lime'),(predictions[i],'red')]:
            hom=joints@sample['K'].numpy().T;uv=hom[:,:2]/np.maximum(hom[:,2:],1e-8)
            for finger in range(5):
                chain=[0]+list(range(1+4*finger,5+4*finger));xy=[tuple(map(float,uv[j])) for j in chain]
                if np.isfinite(uv).all() and np.abs(uv).max()<1e6:pen.line(xy,fill=color,width=2)
        x=slot%4*256;y=slot//4*280;canvas.paste(im,(x,y))
        draw.text((x+3,y+258),f"sample {i}; GT green / prediction red",fill='black')
    dest.parent.mkdir(parents=True,exist_ok=True);canvas.save(dest)


@torch.inference_mode()
def main(config_path,pretrained=False):
    config=json.loads(Path(config_path).read_text());out=Path(config['output_root'])
    name='pre_rgb3' if pretrained else config['name'];assert not pretrained or config['condition']=='rgb3'
    assert int(os.environ['EVENTPOSE_PHYSICAL_GPU'])==0 and torch.cuda.device_count()==1
    torch.set_num_threads(2);device=torch.device('cuda',0)
    checkpoint=None if pretrained else out/'checkpoints'/name/'best.pt'
    checkpoint_sha=(json.loads((out/'pretrained_manifest.json').read_text())['checkpoint_sha256']
                    if pretrained else file_sha256(checkpoint))
    model=load_model(config,device,checkpoint).eval()
    manifest=json.loads((out/'manifest.json').read_text());split=json.loads((out/'split_snapshot.json').read_text())
    results=[]
    for session in manifest['test_sessions']:
        folder=out/'results'/name/session;complete=folder/'complete.json'
        if complete.exists():
            previous=json.loads(complete.read_text());assert previous['checkpoint_sha256']==checkpoint_sha
            results.append(previous);continue
        with h5py.File(out/'cache'/(session+'.h5'),'r',locking=False) as f:
            n=len(f['frame']);target=f['joints'][:];frames=f['frame'][:];people=f['person'][:];right=f['right'][:]
        dataset=HandFrames(config,'test',records=[dict(session=session,row=i) for i in range(n)])
        loader=DataLoader(dataset,batch_size=config['batch_size'],num_workers=config['workers'],pin_memory=True)
        predicted=np.full_like(target,np.nan);seen=np.zeros(n,bool);torch.cuda.synchronize();started=time.perf_counter()
        for batch in loader:
            ids=batch['index'].numpy();batch=to_device(batch,device)
            with torch.autocast('cuda',dtype=torch.float16):pred=model(model_inputs(batch))['joints']
            predicted[ids]=pred.float().cpu().numpy();seen[ids]=True
        torch.cuda.synchronize();seconds=time.perf_counter()-started
        assert seen.all() and np.isfinite(predicted).all()
        values,keys,excluded=score_session(predicted,target,frames,people,right,split['test'][session])
        assert len(values)>0,('No valid original-frame0 hand sequence',session)
        folder.mkdir(parents=True,exist_ok=True)
        temporary=folder/'predictions.partial.npz'
        np.savez_compressed(temporary,pred_joints_cam_m=predicted,target_joints_cam_m=target,
            frame=frames,person=people,right=right,errors_m=values,metric_keys=keys)
        temporary.replace(folder/'predictions.npz')
        if not results:preview(dataset,predicted,out/'visualizations'/(name+'.png'))
        result=dict(session=session,condition=name,checkpoint_sha256=checkpoint_sha,
            processed_samples=n,scored_samples=len(values),finite_predictions=int(np.isfinite(predicted).all((1,2)).sum()),
            excluded_no_frame0=excluded,sums_m=values.sum(0).tolist(),means_mm=(values.mean(0)*1000).tolist(),
            inference_seconds=seconds,samples_per_second=n/seconds,full_test_intervals=True,
            crop_protocol='oracle_gt_device3_hand_boxes',test_used_for_selection=False,
            root_position_error_computed=False,completed_unix=time.time())
        write(complete,result);results.append(result)
        print(json.dumps(dict(session=session,condition=name,samples=n,scored=len(values))),flush=True)
    write(out/'results'/name/'complete.json',dict(condition=name,sessions=len(results),full_test=True,
        checkpoint_sha256=checkpoint_sha,completed_unix=time.time(),root_position_error_computed=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--pretrained',action='store_true')
    a=p.parse_args();main(a.config,a.pretrained)
