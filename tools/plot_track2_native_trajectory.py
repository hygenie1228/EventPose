"""Plot sampled native MHR camera trajectories on Train validation only."""
from pathlib import Path
import json,sys
import numpy as np,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from multihmr.track2_data import OUTPUT,SOURCE,ASSETS,decode_mhr,save_json
from utils.v2d_official_metric import MHR_TABLE3_BODY_JOINT_INDICES


def main():
    torch.set_num_threads(2)
    manifest=json.loads((OUTPUT/'manifest.json').read_text())
    body=torch.jit.load(str(ASSETS/'mhr/mhr_model.pt'),map_location='cpu').eval()
    out=OUTPUT/'native_trajectory_diagnostic';out.mkdir(exist_ok=True)
    fig,axes=plt.subplots(4,2,figsize=(13,13),constrained_layout=True)
    rows={};colors={'GT':'#222222','init_v6':'#1772b6','MultiHMR_RGB':'#d55e00'}
    for row,sample in enumerate(manifest['validation']):
        record=manifest['samples'][sample];frames=np.unique(np.linspace(0,record['frames']-1,64,dtype=int))
        paths={'GT':SOURCE/sample/'human/human_params_gt.npz','init_v6':SOURCE/sample/'human/human_params_init.npz',
            'MultiHMR_RGB':OUTPUT/'predictions/diagnostic_original_native_rgb'/sample/'human_params_mhr_v2.npz'}
        roots={}
        for label,path in paths.items():
            with np.load(path) as z:
                assert len(z['pose'])==record['frames']
                pose=torch.tensor(z['pose'][frames],dtype=torch.float32);shape=torch.tensor(z['shape'],dtype=torch.float32);scales=torch.tensor(z['scales'],dtype=torch.float32)
            values=[]
            with torch.no_grad():
                for start in range(0,len(frames),8):
                    _,joints=decode_mhr(body,pose[start:start+8],shape,scales)
                    values.append(joints[:,MHR_TABLE3_BODY_JOINT_INDICES[0]].numpy())
            roots[label]=np.concatenate(values)
            if label=='GT':
                E=np.asarray(record['world_to_camera']);roots[label]=roots[label]@E[:3,:3].T+E[:3,3]
        t=frames/record['fps'];summaries={};gt_delta=roots['GT']-roots['GT'][0]
        for label,xyz in roots.items():
            delta=xyz-xyz[0]
            axes[row,0].plot(t,xyz[:,2],label=label,color=colors[label],linewidth=1.7)
            axes[row,1].plot(t,np.linalg.norm(delta,axis=1),label=label,color=colors[label],linewidth=1.7)
            denom=float(gt_delta[:,2]@gt_delta[:,2])
            summaries[label]=dict(camera_root_Z_change_gain=float(gt_delta[:,2]@delta[:,2]/denom) if denom>.0001 else None,
                camera_frame0_relative_motion_RMSE_cm=float(np.sqrt(np.mean(np.sum((delta-gt_delta)**2,axis=1)))*100))
        for col in [0,1]:
            axes[row,col].grid(alpha=.25);axes[row,col].set_xlabel('Time (s)');axes[row,col].set_ylabel('Metres')
            axes[row,col].set_title(sample+'\n'+('Camera root depth Z' if col==0 else '3D root displacement from frame 0'),fontsize=9)
        if row==0:axes[row,0].legend(fontsize=9)
        np.savez_compressed(out/(sample+'.npz'),frames=frames,fps=record['fps'],**roots)
        rows[sample]=dict(frames_sampled=frames.tolist(),summary=summaries)
    fig.suptitle('Train-only trajectory diagnosis: original RGB, native MHR root\nCamera coordinates, no Sim3 alignment; 64 sampled original frames per clip',fontsize=13)
    figure=out/'trajectory.png';fig.savefig(figure,dpi=160);plt.close(fig)
    save_json(out/'analysis.json',dict(samples=rows,figure=str(figure),test_used=False,
        GT_for_visual_diagnosis_only=True,GT_world_to_camera_applied_once=True,
        points='Same official MHR body22 pelvis joint for all three conditions',
        not_official_metric=True,not_used_for_training_or_selection=True))
    print('trajectory_plot_ready',figure,flush=True)


if __name__=='__main__':main()
