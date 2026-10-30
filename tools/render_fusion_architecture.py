"""Export implementation-grounded vector figures for the body fusion report."""
import hashlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timedelta, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from goal_runtime_control import register_job


def main():
    goal=json.loads((ROOT/'_agents/research/261027/device34_fusion_repair_goal_15h.json').read_text())
    register_job(goal, os.getpid(), 'fusion_repair_architecture_figures')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':12,
                         'pdf.fonttype':42, 'svg.fonttype':'none'})
    out=Path(goal['output_root'])/'figures';out.mkdir(exist_ok=True)
    C={'base':'#e9f1fb','learn':'#fff0dc','data':'#f2f5f8',
       'gt':'#f0eafa','out':'#e8f5ec','edge':'#334155','muted':'#526174'}
    outputs=[]

    def canvas(title, subtitle, height=66):
        fig,ax=plt.subplots(figsize=(24, height*.25))
        fig.subplots_adjust(left=.015,right=.985,bottom=.02,top=.98)
        ax.set(xlim=(0,100),ylim=(0,height));ax.axis('off')
        ax.text(1,height-1,title,fontsize=24,weight='bold',va='top',color=C['edge'])
        ax.text(1,height-4,subtitle,fontsize=12,va='top',color=C['muted'])
        return fig,ax

    def box(ax,x,y,w,h,title,body,kind='base',fs=12):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.25,rounding_size=.65',
                    fc=C[kind],ec='#94a3b8',lw=1.15))
        ax.text(x+w/2,y+h-1.15,title,ha='center',va='top',fontsize=14,weight='bold',color=C['edge'])
        ax.text(x+w/2,y+h/2-.8,body,ha='center',va='center',fontsize=fs,
                linespacing=1.5,color=C['edge'])

    def arrow(ax,a,b,label=None,dashed=False,curve=None):
        kw=dict(arrowstyle='-|>',mutation_scale=17,lw=1.6,color=C['edge'],
                linestyle='--' if dashed else '-')
        if curve:kw['connectionstyle']=curve
        ax.add_patch(FancyArrowPatch(a,b,**kw))
        if label:ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.6,label,ha='center',va='bottom',
                         fontsize=10,color=C['muted'],bbox=dict(fc='white',ec='none',pad=1))

    def save(fig,name):
        for ext in ('svg','pdf','png'):
            path=out/(name+'.'+ext);fig.savefig(path,dpi=170,facecolor='white')
            outputs.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        plt.close(fig)

    fig,ax=canvas('A  |  Calibration-aware RGB / event body reconstruction',
        'Current implementation: epipolar_repair_norm_v2  •  Per-frame Multi-HMR  •  No temporal network or GT input at inference')
    ax.text(1,58,'SENSOR ROUTING',weight='bold',fontsize=12,color=C['muted'])
    ax.text(17,58,'RGB3 + EV4: primary RGB3 / auxiliary EV4    |    EV3 + RGB4: primary RGB4 / auxiliary EV3',fontsize=12)
    ax.text(17,55.6,'EV3 + EV4: primary EV3 / auxiliary EV4    |    RGB controls: same RGB path, auxiliary branch omitted',fontsize=12)
    box(ax,1,39,16,13,'Primary observation',
        'RGB: 3 × 896 × 896\nor event: 2 × 224 × 224\nFull sensor frame\nCorrected past timestamp',kind='data')
    box(ax,21,39,16,13,'Primary input path',
        'RGB: ImageNet normalize\nEvent: 1×1 projection +\nzero-init spatial refiner\nResize to 3 × 896 × 896',kind='learn')
    box(ax,41,39,16,13,'DINOv2 ViT-L/14',
        'Verified base warmstart\n64 × 64 spatial tokens\nF: B × 4096 × 1024\n896-pixel primary K',kind='base')
    box(ax,61,39,16,13,'Residual feature fusion',
        'F\u2032 = F + available · ΔF\nΔF initially exactly zero\nMissing auxiliary: F\u2032 = F\nShared native head input',kind='learn')
    box(ax,81,39,17,13,'Native Multi-HMR head',
        'Human detection + NMS\nCamera-ray embedding (K)\nHPH cross-attention\nSMPL-X reconstruction',kind='base')
    for x in [17,37,57,77]:arrow(ax,(x,45.5),(x+4,45.5))
    box(ax,1,19,16,13,'Auxiliary event input',
        '2 × 224 × 224\nPolarity-split count33\nPast 33 ms observations\nNo RGB in event-only',kind='data')
    box(ax,21,19,16,13,'Independent event CNN',
        'Input gain = 4\nConv / GroupNorm / GELU\n3 stride-2 stages\nE: B × 128 × 28 × 28',kind='learn')
    box(ax,41,19,16,13,'Calibrated ray sampling',
        '16 × 16 primary queries\n11 camera-Z hypotheses\nProject using K, R, t\nSample E along each ray',kind='base')
    box(ax,61,19,16,13,'Query-conditioned message',
        'Masked attention over Z\nLinear output: zero init\n16 × 16 → 64 × 64\nΔF: B × 4096 × 1024',kind='learn')
    box(ax,81,19,17,13,'Per-person outputs',
        'Score + 2D location\nPose / shape / expression\n3D mesh + pelvis translation\n55 regressed joints; score 22',kind='out')
    for x in [17,37,57]:arrow(ax,(x,25.5),(x+4,25.5))
    arrow(ax,(49,39),(49,32),'query features')
    arrow(ax,(69,32),(69,39),'ΔF')
    arrow(ax,(89.5,39),(89.5,32),'native prediction')
    box(ax,1,2,36,12,'Calibration / causal correspondence',
        'Separate RGB and event pixel planes\nKp, Ka and primary → auxiliary R, t\nNo assumed same-pixel alignment\nZ hypotheses = 0.5 ... 8 m; not measured GT depth',kind='data')
    ax.plot([37,39,39,49],[10,10,17,17],color=C['edge'],lw=1.6)
    arrow(ax,(49,17),(49,19))
    box(ax,41,2,57,12,'Inference boundary',
        'Inputs: sensor pixels + camera calibration + availability only\nGT pose / box / depth never enters inference or event-only detection\nTraining only: GT center indices supervise human queries and losses\nNo stereo triangulation, depth sensor input, or recurrent temporal module',kind='gt',fs=13)
    save(fig,'신체센서융합_모델구조')

    fig,ax=canvas('B  |  Data, training and held-out evaluation protocol',
        'Same split, fixed subset, batch and training time across five repaired / control conditions',height=62)
    box(ax,1,41,17,13,'Read-only source data',
        'Body sessions: suffix _b\nRGB video + sparse events\nCorrected timestamps\nPer-sensor calibration',kind='data')
    box(ax,22,41,17,13,'Frozen input mapping',
        'Past-only correspondence\nFull-frame RGB letterbox\ncount33 event encoding\nSHA256 + frame validation',kind='data')
    box(ax,43,41,17,13,'Disjoint Train / holdout',
        'Train: 51 sessions\nFixed 6,501 frames\nHoldout: 9 sessions / 288\nSame inputs for five models',kind='data')
    box(ax,64,41,16,13,'Fine-tuning',
        'Verified warmstart\nAdamW / AMP / batch16\n30 min head + adapter first\nThen last 8 ViT blocks',kind='learn')
    box(ax,84,41,14,13,'Select checkpoint',
        '3 hours / condition\nTrain holdout only\nCoverage ≥ 95% first\nThen lowest PA error',kind='out',fs=11)
    for a,b in [(18,22),(39,43),(60,64),(80,84)]:arrow(ax,(a,47.5),(b,47.5))
    box(ax,1,21,17,13,'Train GT supervision',
        'Visible person centers\n22-body-joint targets\nRoot / rotations / shape\nGT indices: training only',kind='gt')
    box(ax,22,21,38,13,'Training objectives',
        '100 LPA + 25 Lpelvis-aligned MPJPE + Ldetection + Loffset\n+ 0.2 Lrotation + 0.02 Lshape + Labs-root + 0.01 Lresidual\nLoss geometry in meters; reported pose errors in millimeters\nFinite-gradient checks, clipping and periodic checkpoints',kind='learn',fs=12)
    arrow(ax,(18,27.5),(22,27.5),dashed=True)
    ax.plot([60,62,62,72],[27.5,27.5,37,37],color=C['edge'],lw=1.6,ls='--')
    ax.text(67,37.6,'Train only',ha='center',va='bottom',fontsize=10,color=C['muted'])
    arrow(ax,(72,37),(72,41),dashed=True)
    box(ax,64,21,34,13,'Full native Test inference',
        'Test pixels + calibration via the same frozen input mapping\n18 body sessions / subjects; 80 × 10-second chunks\n24,080 frames; no GT-selected crop or detection\nPrimary camera → common RGB3; GT only for scoring',kind='out')
    arrow(ax,(91,41),(91,34))
    box(ax,1,1,59,14,'Five primary metrics on the identical common GT population',
        'First-frame MPJPE / PA-MPJPE: align exactly at frame 0, then freeze\nFrame-aligned MPJPE / PA-MPJPE: recompute alignment per frame\nMPJPE: pelvis translation; PA: proper Sim(3), reflection forbidden\nRoot position error = mean ||rootpred − rootGT||₂ × 1000 (mm)',kind='out',fs=13)
    box(ax,64,1,34,14,'Independent reporting / verification',
        'Misses remain misses; detection coverage separately\nRPE-Z, absolute pose and motion errors separately\nCluster CI / Holm; actual prediction visualization\nSame-GPU online / cache speed + raw metric audit',kind='out',fs=12)
    arrow(ax,(81,21),(81,15))
    arrow(ax,(64,8),(60,8))
    save(fig,'신체센서융합_학습평가_파이프라인')
    sources=['src/multihmr/multisensor_repair.py','src/multihmr/multisensor_model.py',
             'src/multihmr/multisensor_data.py','src/multihmr/multisensor_preparation.py',
             'src/utils/event_encoding.py','src/multihmr/model.py','scripts/train_multihmr.py',
             'src/multihmr/multisensor_inference.py','tools/analyze_fusion_repair.py']
    record=dict(created_kst=datetime.now(timezone(timedelta(hours=9))).isoformat(),
                outputs=outputs,sources={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in sources},
                architecture='epipolar_repair_norm_v2',
                scope='Implementation diagram, not experimental performance evidence')
    (out/'모델구조_도식_출처.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(figures=2,formats=['svg','pdf','png'],folder=str(out)),ensure_ascii=False))


if __name__=='__main__':main()
