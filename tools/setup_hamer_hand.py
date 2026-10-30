"""Extract only official HaMeR assets, retaining source/checksum provenance."""
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
OUT=ROOT/'exp/261028/hamer_hand'
URL='https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz'


def main():
    import requests
    from hamer.event_model import file_sha256
    final=OUT/'assets/hamer_demo_data.tar.gz';partial=final.with_name(final.name+'.partial')
    if not final.exists():
        response=requests.head(URL,allow_redirects=True,timeout=60);response.raise_for_status()
        size=int(response.headers['Content-Length'])
        import psutil
        import subprocess
        from run_hamer_hand import register
        download=None;registered=set()
        while not partial.exists() or partial.stat().st_size<size:
            owners=[]
            for process in psutil.process_iter(['pid','cmdline']):
                try:
                    args=process.info['cmdline'] or []
                    if args and Path(args[0]).name=='curl' and URL in args and any(
                            Path(arg).resolve()==partial.resolve() for arg in args if 'hamer_demo_data.tar.gz.partial' in arg):
                        owners.append(process.info['pid'])
                except (psutil.NoSuchProcess,psutil.AccessDenied):pass
            if not owners:
                if download is not None:assert download.poll()==0,'Official pretrained download failed'
                download=subprocess.Popen(['curl','-L','--fail','--retry','3','--continue-at','-',
                    '-o',str(partial),URL],cwd=ROOT)
                owners=[download.pid]
            for pid in owners:
                if pid not in registered:register(pid,'official_pretrained_download');registered.add(pid)
            time.sleep(30)
        assert partial.stat().st_size==size
        partial.replace(final)
    selected={};members=[]
    with tarfile.open(final,'r:gz') as archive:
        for member in archive:
            base=Path(member.name).name
            if base not in ('hamer.ckpt','model_config.yaml','mano_mean_params.npz'):continue
            dest=OUT/'assets'/base
            if not dest.exists() or dest.stat().st_size!=member.size:
                reader=archive.extractfile(member);assert reader is not None
                tmp=dest.with_name(dest.name+'.partial')
                with tmp.open('wb') as writer:shutil.copyfileobj(reader,writer,8*1024*1024)
                assert tmp.stat().st_size==member.size;tmp.replace(dest)
            selected[base]=str(dest);members.append(dict(name=member.name,size=member.size))
    assert set(selected)=={'hamer.ckpt','model_config.yaml','mano_mean_params.npz'},selected
    manifest=dict(source_url=URL,archive=str(final),archive_sha256=file_sha256(final),
        checkpoint=selected['hamer.ckpt'],checkpoint_sha256=file_sha256(selected['hamer.ckpt']),
        config=selected['model_config.yaml'],config_sha256=file_sha256(selected['model_config.yaml']),
        mean_params=selected['mano_mean_params.npz'],mean_params_sha256=file_sha256(selected['mano_mean_params.npz']),
        extracted_members=members,upstream=json.loads((ROOT/'_agents/research/261028/hamer_upstream.json').read_text()),
        licensed_MANO_right_sha256=file_sha256(ROOT/'data/human_model_files/mano/MANO_RIGHT.pkl'),
        original_dataset_updates=0,downloaded_unix=time.time(),storage='experiment assets; data kept read-only')
    from prepare_hamer_hand import write
    write(OUT/'pretrained_manifest.json',manifest)
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('upstream',)}),flush=True)


if __name__=='__main__':main()
