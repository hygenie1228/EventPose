"""Compare native init trajectory diagnostics on the four Train holdouts."""
from pathlib import Path
import json,sys,time
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from multihmr.track2_data import OUTPUT,SOURCE,ASSETS,decode_mhr,save_json,sha256
from utils import v2d_official_metric as official


def main():
    torch.set_num_threads(2)
    manifest=json.loads((OUTPUT/'manifest.json').read_text());body=torch.jit.load(str(ASSETS/'mhr/mhr_model.pt'),map_location='cpu').eval()
    rows={};dest=OUTPUT/'init_v6_Train_trajectory_diagnostic.json'
    for sample in manifest['validation']:
        roles=json.loads((OUTPUT/'predictions/original'/sample/'metrics.json').read_text())['protocol']['roles']
        selected=roles['alignment']+roles['body'];points={}
        for label,filename in [('GT','human_params_gt.npz'),('init_v6','human_params_init.npz')]:
            with np.load(SOURCE/sample/'human'/filename) as z:
                pose=torch.tensor(z['pose'],dtype=torch.float32);shape=torch.tensor(z['shape'],dtype=torch.float32);scales=torch.tensor(z['scales'],dtype=torch.float32)
            values=[]
            with torch.no_grad():
                for start in range(0,len(pose),8):
                    v,_=decode_mhr(body,pose[start:start+8],shape,scales);values.append(v[:,selected].numpy())
            points[label]=np.concatenate(values).astype(np.float64)
        E=np.asarray(manifest['samples'][sample]['world_to_camera'])
        target=points['GT']@E[:3,:3].T+E[:3,3];pred=points['init_v6']
        transform=official.fit_similarity_transform(pred[0,:27],target[0,:27]);aligned=official.apply_similarity_transform(pred,transform)
        shift=target[:,:27].mean(1)-aligned[:,:27].mean(1)
        cd=lambda a:float(official.symmetric_chamfer_distance(a[:,27:],target[:,27:]).mean()*100)
        per_frame=np.stack([official.apply_similarity_transform(pred[i],official.fit_similarity_transform(pred[i,:27],target[i,:27])) for i in range(len(pred))])
        rows[sample]=dict(frames=len(pred),MHR_fixed_frame0_CD_H_cm=cd(aligned),
            diagnostic_per_frame_center_corrected_CD_cm=cd(aligned+shift[:,None]),
            diagnostic_per_frame_Sim3_CD_cm=cd(per_frame),
            fixed_alignment_centroid_trajectory_RMSE_cm=float(np.sqrt(np.mean(np.sum(shift**2,axis=1)))*100),
            frame0_Sim3_scale=float(transform[0]),init_sha256=sha256(SOURCE/sample/'human/human_params_init.npz'))
        save_json(dest,dict(samples=rows,test_used=False,all_four_complete=False,GT_for_diagnostics_only=True,updated_unix=time.time()))
        print(sample,json.dumps(rows[sample]),flush=True)
    summary={k:float(np.mean([v[k] for v in rows.values()])) for k in ['MHR_fixed_frame0_CD_H_cm','diagnostic_per_frame_center_corrected_CD_cm','diagnostic_per_frame_Sim3_CD_cm','fixed_alignment_centroid_trajectory_RMSE_cm']}
    save_json(dest,dict(samples=rows,summary=summary,test_used=False,all_four_complete=True,
        GT_for_diagnostics_only=True,caveat='Only fixed-frame0 MHR CD-H is the final metric; alternative alignments are non-additive diagnostics.',updated_unix=time.time()))


if __name__=='__main__':main()
