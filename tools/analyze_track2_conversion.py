"""CPU, prediction-only MHR conversion diagnostic on the Train holdout clips."""
from pathlib import Path
import json
import sys
import time

import numpy as np
import torch
import smplx

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from multihmr.track2_data import OUTPUT,ASSETS,decode_mhr,mapping,transfer,save_json,sha256
from utils import v2d_official_metric as metric


def main():
    torch.set_num_threads(2)
    manifest=json.loads((OUTPUT/'manifest.json').read_text())
    roles=json.loads((OUTPUT/'body_roles.json').read_text())['roles']
    selected=roles['alignment']+roles['body']
    model=torch.jit.load(str(ASSETS/'mhr/mhr_model.pt'),map_location='cpu').eval()
    body=smplx.create(str(ASSETS),model_type='smplx',gender='neutral',use_pca=False,flat_hand_mean=True,num_betas=10)
    ids,weights=mapping('smplx2mhr',body.faces,'cpu');ids=ids[selected];weights=weights[selected]
    results={}
    with torch.no_grad():
        for name in ['original','search_head_strong','search_last2_strong','search_last4_strong']:
            values={}
            for sample in manifest['validation']:
                folder=OUTPUT/'predictions'/name/sample
                prediction=folder/'human_params_mhr.npz';smpl=folder/'smplx_predictions.npz'
                if not prediction.exists() or not smpl.exists():continue
                with np.load(prediction) as z:
                    pose=z['pose'];shape=z['shape'];scales=z['scales'];errors=z['per_frame_vertex_error_mm']
                frames=np.unique(np.concatenate([np.linspace(0,len(pose)-1,min(32,len(pose)),dtype=int),np.argsort(errors)[-5:]]))
                with np.load(smpl) as z:source=transfer(torch.tensor(z['vertices_cam_m'][frames]),ids,weights).numpy()
                converted=[]
                for start in range(0,len(frames),8):
                    v,_=decode_mhr(model,torch.tensor(pose[frames[start:start+8]]),torch.tensor(shape),torch.tensor(scales))
                    converted.append(v[:,selected].numpy())
                converted=np.concatenate(converted).astype(np.float64);source=source.astype(np.float64)
                aligned=metric.apply_similarity_transform(converted[:,27:],metric.fit_similarity_transform(converted[0,:27],source[0,:27]))
                before=metric.symmetric_chamfer_distance(converted[:,27:],source[:,27:])*100
                after=metric.symmetric_chamfer_distance(aligned,source[:,27:])*100
                corresponding=np.linalg.norm(converted[:,27:]-source[:,27:],axis=-1).mean(1)*100
                values[sample]=dict(sampled_frames=frames.tolist(),conversion_mean_body_correspondence_error_cm=float(corresponding.mean()),
                    conversion_body_chamfer_unaligned_cm=float(before.mean()),conversion_body_chamfer_frame0_aligned_cm=float(after.mean()),
                    max_sampled_frame_correspondence_error_cm=float(corresponding.max()),
                    prediction_sha256=sha256(prediction),smplx_source_sha256=sha256(smpl),
                    full_mesh_mean_fit_mm=float(errors.mean()),full_mesh_max_fit_mm=float(errors.max()))
                print(name,sample,'conversion-only frame0 CD cm',round(float(after.mean()),4),flush=True)
            if values:
                keys=['conversion_mean_body_correspondence_error_cm','conversion_body_chamfer_unaligned_cm','conversion_body_chamfer_frame0_aligned_cm']
                results[name]=dict(sequences=len(values),per_sequence=values,**{key:float(np.mean([v[key] for v in values.values()])) for key in keys})
            save_json(OUTPUT/'conversion_metric_diagnostic.json',dict(models=results,GT_read=False,gpu_used=False,
                scope='Four Train-validation clips, 32 uniform frames plus five largest prediction-only conversion residuals; not full-frame scores',
                not_human_prediction_accuracy=True,includes_mapping_and_MHR_fitting_error=True,created_unix=time.time()))


if __name__=='__main__':main()
