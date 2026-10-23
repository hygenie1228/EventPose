"""Remove masks and repack the five processed labels, keeping source datasets intact."""
from pathlib import Path
import json
import shutil
import h5py
import numpy as np

ROOT=Path(__file__).absolute().parents[1]
SESSION=ROOT/'etri_processed/20260720_125338_040324_200_f'
WORK=ROOT/'exp/261023/processed_storage_audit'
GROUPS=('params','bboxes','frames')


def main():
    WORK.mkdir(exist_ok=True,parents=True);records=[]
    for i in range(5):
        path=SESSION/f'device{i}/label.h5'
        assert path.resolve().is_relative_to((ROOT/'etri_processed').resolve())
        before=path.stat().st_size
        staged=WORK/f'device{i}_label_no_masks.h5'
        with h5py.File(path,'r') as src,h5py.File(staged,'w') as dst:
            for key in GROUPS:src.copy(key,dst)
            for key,value in src.attrs.items():
                if 'mask' not in key:dst.attrs[key]=value
            dst.attrs['person_masks_stored']=False
        with h5py.File(path,'r') as src,h5py.File(staged,'r') as dst:
            assert set(dst)==set(GROUPS)
            def check(name,obj):
                if isinstance(obj,h5py.Dataset):
                    assert obj.shape==dst[name].shape and obj.dtype==dst[name].dtype
                    assert np.array_equal(obj[:],dst[name][:]),name
            for group in GROUPS:src[group].visititems(lambda n,o:check(group+'/'+n,o))
        # Stage is on another filesystem. Copy back after closing and validating it.
        shutil.copyfile(staged,path)
        with h5py.File(path,'r') as dst:assert set(dst)==set(GROUPS)
        records.append({'device':f'device{i}','before_bytes':before,'after_bytes':path.stat().st_size,'groups':list(GROUPS)})
        staged.unlink()
        print(records[-1],flush=True)
    (WORK/'mask_removal_validation.json').write_text(json.dumps(records,indent=2)+'\n')


if __name__=='__main__':main()
