"""Prepare only Train20 fixed RGB viewports from supplied mask videos on CPU."""
from pathlib import Path
import argparse,fcntl,json,os,sys,time
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from multihmr.track2_data import OUTPUT,SOURCE,save_json,sha256
from multihmr.target_crop import fixed_target_box


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--split',choices=['train','test'],default='train');args=parser.parse_args()
    prefix='fixed_view_training_preparation' if args.split=='train' else 'fixed_view_Test_preparation'
    cv2.setNumThreads(1)
    with (OUTPUT/'locks'/(prefix+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest=json.loads((OUTPUT/'manifest.json').read_text());records={}
        control=json.loads((OUTPUT/'static_crop_Train_control_plan.json').read_text())
        destination=OUTPUT/('fixed_view_training_plan.json' if args.split=='train' else 'fixed_view_Test_plan.json')
        if destination.exists():
            value=json.loads(destination.read_text());assert set(value['samples'])==set(manifest[args.split]);return
        cache=OUTPUT/'fixed_view_mask_bounds';cache.mkdir(exist_ok=True)
        for sample in manifest[args.split]:
            if args.split=='train':assert sample not in manifest['test']
            else:assert (OUTPUT/'fixed_view_adaptation_plan.json').exists(),'Freeze Train-derived recipe before preparing Test inputs'
            r=manifest['samples'][sample]
            path=SOURCE/sample/'processed/human_mask.mp4';digest=sha256(path);cached=cache/(sample+'.json')
            if cached.exists():
                entry=json.loads(cached.read_text());assert entry['mask_sha256']==digest
            elif sample in control['samples']:
                entry=control['samples'][sample];assert entry['mask_sha256']==digest
            else:
                cap=cv2.VideoCapture(str(path));assert cap.isOpened();frames=0
                bounds=np.array([r['width'],r['height'],0,0])
                while True:
                    ok,mask=cap.read()
                    if not ok:break
                    gray=cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
                    if gray.shape!=(r['height'],r['width']):gray=cv2.resize(gray,(r['width'],r['height']),interpolation=cv2.INTER_NEAREST)
                    yy,xx=np.where(gray>127)
                    if len(xx):bounds=np.array([min(bounds[0],xx.min()),min(bounds[1],yy.min()),max(bounds[2],xx.max()+1),max(bounds[3],yy.max()+1)])
                    frames+=1
                cap.release();assert frames==r['frames']
                entry=dict(box=fixed_target_box(bounds,[r['height'],r['width']],r['K']),mask_union=bounds.tolist(),
                    frames=frames,mask_sha256=digest,GT_used=False)
            assert entry['box']==fixed_target_box(entry['mask_union'],[r['height'],r['width']],r['K'])
            assert entry['frames']==r['frames'];save_json(cached,entry);records[sample]=entry
            save_json(OUTPUT/(prefix+'_status.json'),dict(status='preparing_'+args.split+'_input_geometry_only',
                pid=os.getpid(),completed=list(records),GT_read=False,test_labels_or_scores_used=False,updated_unix=time.time()))
            print(sample,entry['box'],flush=True)
        assert set(records)==set(manifest[args.split])
        save_json(destination,dict(samples=records,policy='fixed_mask_union_view_v1',reference_fov_deg=60,
            margin=.1,target_mask_only_for_viewport_and_output_selection=True,background_RGB_retained=True,
            dynamic_zoom=False,K_transformed=True,GT_read=False,test_labels_or_scores_used=False,created_unix=time.time()))
        save_json(OUTPUT/(prefix+'_status.json'),dict(status='complete',pid=os.getpid(),
            completed=list(records),GT_read=False,test_labels_or_scores_used=False,updated_unix=time.time()))
        train_path=OUTPUT/'fixed_view_training_plan.json';test_path=OUTPUT/'fixed_view_Test_plan.json'
        if train_path.exists() and test_path.exists():
            train=json.loads(train_path.read_text());test=json.loads(test_path.read_text());combined=dict(train)
            combined['samples']=dict(train['samples'],**test['samples']);combined['test_labels_or_scores_used']=False
            combined['scope']='Frozen Train-derived rule applied to all30 provided input mask videos; no GT parameters or model scores'
            assert set(combined['samples'])==set(manifest['samples']);save_json(OUTPUT/'fixed_view_all30_plan.json',combined)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        save_json(OUTPUT/'fixed_view_training_preparation_error.json',dict(error=repr(error),time=time.time()));raise
