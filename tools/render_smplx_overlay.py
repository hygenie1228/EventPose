"""Full-frame video | SMPL-X overlay using processed data and local model assets only."""
import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import cv2
import h5py
import numpy as np

ROOT=Path(__file__).absolute().parents[1]
SESSION=ROOT/'etri_processed/20260720_125338_040324_200_f'
OUT=ROOT/'exp/261023/smplx_overlay'
WORK=OUT/'_work'
KEYS=('smplx_pose_axis_angle','smplx_betas','translation_world_m')


def build_cache():
    from visualize_processed_session import body_model,vertices
    WORK.mkdir(parents=True,exist_ok=True)
    with h5py.File(SESSION/'device0/label.h5','r') as f:
        params={k:f['params/'+k][:] for k in KEYS}
        times=f['frames/reference_timestamp_ns'][:].astype(np.int64)
    model_path=ROOT/'data/human_model_files/smplx/SMPLX_NEUTRAL.npz'
    model_hash=hashlib.sha256(model_path.read_bytes()).hexdigest()
    digest=hashlib.sha256()
    for k in KEYS:digest.update(params[k].tobytes())
    signature={'model_sha256':model_hash,'params_sha256':digest.hexdigest(),'frames':len(times)}
    cache_path=WORK/'world_vertices.npy';meta_path=WORK/'cache.json'
    if all(p.exists() for p in [cache_path,meta_path,WORK/'reference_timestamps_ns.npy',WORK/'faces.npy']) and json.loads(meta_path.read_text())==signature:
        print('Reusing verified common world mesh cache',flush=True)
        return
    model=body_model();n,people=params['smplx_betas'].shape[:2]
    cache=np.lib.format.open_memmap(cache_path,mode='w+',dtype=np.float32,shape=(n,people,10475,3))
    for a in range(0,n,32):
        b=min(a+32,n)
        v=vertices(model,params[KEYS[0]][a:b].reshape(-1,165),params[KEYS[1]][a:b].reshape(-1,16),params[KEYS[2]][a:b].reshape(-1,3))
        assert np.isfinite(v).all()
        cache[a:b]=v.reshape(b-a,people,10475,3)
        if a%512==0:print('Reconstructing SMPL-X world meshes',a,'/',n,flush=True)
    cache.flush();del cache
    np.save(WORK/'reference_timestamps_ns.npy',times)
    np.save(WORK/'faces.npy',model.faces.astype(np.int32))
    meta_path.write_text(json.dumps(signature,indent=2)+'\n')
    print('All world meshes reconstructed from params',flush=True)


class Renderer:
    def __init__(self,width,height,K,gpu):
        os.environ['EGL_DEVICE_ID']=str(gpu)
        import pyrender
        self.pr=pyrender
        self.renderer=pyrender.OffscreenRenderer(width,height)
        self.renderer._platform.make_current()
        from OpenGL.GL import glGetString,GL_VENDOR,GL_RENDERER
        self.gl_vendor=glGetString(GL_VENDOR).decode()
        self.gl_renderer=glGetString(GL_RENDERER).decode()
        assert 'NVIDIA' in self.gl_vendor,self.gl_vendor
        print('EGL GPU',gpu,self.gl_vendor,self.gl_renderer,flush=True)
        self.scene=pyrender.Scene(bg_color=[0,0,0,0],ambient_light=[.35,.35,.35])
        assert abs(K[0,1])<1e-7
        camera=pyrender.IntrinsicsCamera(float(K[0,0]),float(K[1,1]),float(K[0,2]+.5),float(K[1,2]+.5),znear=.03,zfar=100)
        self.scene.add(camera,pose=np.eye(4))
        self.scene.add(pyrender.DirectionalLight(color=np.ones(3),intensity=2.0),pose=np.eye(4))
        self.material=pyrender.MetallicRoughnessMaterial(baseColorFactor=(.15,.62,.95,1),metallicFactor=0,roughnessFactor=.7)
        self.nodes=[]
        # Check the exact OpenGL-to-image pixel convention against CV projection.
        cvpoints=np.array([[.1,.2,2],[-.3,.1,3],[0,0,1.]])
        glpoints=cvpoints*np.array([1,-1,-1])
        glh=np.c_[glpoints,np.ones(len(glpoints))]
        clip=glh@camera.get_projection_matrix(width,height).T
        ndc=clip[:,:3]/clip[:,3:]
        uv=np.c_[(ndc[:,0]+1)*width/2-.5,(1-ndc[:,1])*height/2-.5]
        projected=cvpoints@K.T;expected=projected[:,:2]/projected[:,2:]
        self.projection_error=float(np.max(np.abs(uv-expected)))
        assert self.projection_error<1e-6,self.projection_error

    def draw(self,v_camera,faces):
        import trimesh
        for node in self.nodes:self.scene.remove_node(node)
        self.nodes=[]
        for v in v_camera:
            vgl=(v*np.array([1,-1,-1])).astype(np.float32)
            mesh=self.pr.Mesh.from_trimesh(trimesh.Trimesh(vertices=vgl,faces=faces,process=False),material=self.material,smooth=True)
            self.nodes.append(self.scene.add(mesh))
        return self.renderer.render(self.scene,flags=self.pr.RenderFlags.RGBA|self.pr.RenderFlags.SKIP_CULL_FACES)

    def close(self):self.renderer.delete()


def render_device(i,gpu,limit=None):
    cv2.setNumThreads(1)
    device=SESSION/f'device{i}'
    source=device/'rgb.mp4'
    if not source.exists():source=device/'video.mp4'
    cap=cv2.VideoCapture(str(source),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,2])
    assert cap.isOpened()
    width,height=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=float(cap.get(cv2.CAP_PROP_FPS))
    rw=384;rh=round(height*rw/width);out_h=rh+(rh%2)
    sx,sy=rw/width,rh/height
    resize=np.array([[sx,0,(sx-1)/2],[0,sy,(sy-1)/2],[0,0,1.]])
    with np.load(device/'event.npz',allow_pickle=False) as z:
        K=z['rgb_calibration__rgb__intrinsic_matrix'];R=z['rgb_calibration__rgb__rotation_matrix'];t=z['rgb_calibration__rgb__translation']
        rgb_times=z['rgb__timestamp_ns'].astype(np.int64)
    assert len(rgb_times)==n and np.allclose(R@R.T,np.eye(3),atol=1e-6)
    with h5py.File(device/'label.h5','r') as f:
        positions=f['frames/source_rgb_position'][:].astype(np.int64)
        label_times=f['frames/source_rgb_timestamp_ns'][:].astype(np.int64)
        times=f['frames/reference_timestamp_ns'][:].astype(np.int64)
        params={k:f['params/'+k][:] for k in KEYS}
    assert np.array_equal(rgb_times[positions],label_times)
    assert len(np.unique(positions))==len(positions)
    ref_times=np.load(WORK/'reference_timestamps_ns.npy')
    lookup=np.searchsorted(ref_times,times);assert np.array_equal(ref_times[lookup],times)
    with h5py.File(SESSION/'device0/label.h5','r') as f:
        for key in KEYS:assert np.array_equal(f['params/'+key][lookup],params[key])
    cache=np.load(WORK/'world_vertices.npy',mmap_mode='r');faces=np.load(WORK/'faces.npy')
    row_for_frame=np.full(n,-1,np.int64);row_for_frame[positions]=np.arange(len(positions))
    renderer=Renderer(rw,rh,resize@K,gpu)
    count=min(n,int(fps*60)) if limit is None else min(n,limit)
    suffix='_smoke' if limit is not None else ''
    output=OUT/f'device{i}_video_smplx{suffix}.mp4'
    writer=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{rw*2}x{out_h}',
        '-r',str(fps),'-i','-','-an','-c:v','libx264','-threads','2','-preset','veryfast','-crf','24',
        '-profile:v','baseline','-level','3.0','-tag:v','avc1',
        '-pix_fmt','yuv420p','-movflags','+faststart',str(output)],stdin=subprocess.PIPE)
    seen=0;covered=[];outside=[];start=time.monotonic()
    snapshots={0,len(positions)//4,len(positions)//2,3*len(positions)//4,len(positions)-1}
    try:
        for frame in range(count):
            ok,image=cap.read();assert ok,(i,frame)
            small=cv2.resize(image,(rw,rh),interpolation=cv2.INTER_AREA)
            over=small.copy();row=int(row_for_frame[frame])
            if row>=0:
                v=np.asarray(cache[int(lookup[row])])@R.T+t
                colors,depth=renderer.draw(v,faces)
                mask=depth>0
                if not mask.any():outside.append(frame)
                mesh_bgr=colors[:,:,:3][:,:,::-1]
                blend=cv2.addWeighted(small,.45,mesh_bgr,.55,0)
                over[mask]=blend[mask]
                seen+=1
                if row in snapshots:
                    covered.append(float(np.mean(mask)))
                    cv2.imwrite(str(WORK/f'device{i}_check_gt{row}.jpg'),np.hstack([small,over]))
            canvas=np.zeros((out_h,rw*2,3),np.uint8)
            canvas[:rh,:rw]=small;canvas[:rh,rw:]=over
            writer.stdin.write(memoryview(canvas))
            if frame%250==0:print(i,'rendered',frame,'/',count,'seconds',round(time.monotonic()-start,1),flush=True)
    finally:
        cap.release();renderer.close();writer.stdin.close();code=writer.wait()
    assert code==0
    assert seen==np.count_nonzero(row_for_frame[:count]>=0) and len(outside)<seen
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=codec_name,pix_fmt,nb_frames,width,height','-show_entries','format=duration','-of','json',str(output)]))
    assert int(probe['streams'][0]['nb_frames'])==count
    report={'device':i,'gpu':gpu,'file':output.name,'frames':count,'GT_poses_rendered':seen,'all_GT_poses_rendered':seen==len(positions),
        'frames_without_GT':np.flatnonzero(row_for_frame[:count]<0).tolist(),'original_video':str(source),
        'model_file':'data/human_model_files/smplx/SMPLX_NEUTRAL.npz','stored_mesh_used':False,
        'data_sources':['label.h5','event.npz',source.name],
        'K_native':K.tolist(),'R_world_to_camera':R.tolist(),'t_world_to_camera_m':t.tolist(),
        'GL_vendor':renderer.gl_vendor,'GL_renderer':renderer.gl_renderer,'mesh_outside_image_frames':outside,
        'native_to_render_pixel_transform':resize.tolist(),'projection_convention_max_error_pixels':renderer.projection_error,
        'video_timestamp_mapping_exact':True,'selected_video_positions':positions.tolist(),
        'GT_anchor_max_residual_ms':float(np.max(np.abs(label_times-times))/1e6),'sample_mesh_coverage':covered,'probe':probe}
    assert float(probe['format']['duration'])<=60
    report['all_clip_GT_poses_rendered']=True
    (WORK/f'device{i}_validation{suffix}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(i,'ALL VIDEO FRAMES AND GT POSES VERIFIED',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',type=int);parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--limit',type=int);parser.add_argument('--build-cache',action='store_true');args=parser.parse_args()
    WORK.mkdir(parents=True,exist_ok=True)
    if args.build_cache:build_cache();return
    if args.device is not None:render_device(args.device,args.gpu,args.limit);return
    build_cache()
    def run(i):
        with (WORK/f'device{i}.log').open('w') as log:
            subprocess.run([sys.executable,str(Path(__file__).absolute()),'--device',str(i),'--gpu',str(i)],stdout=log,stderr=subprocess.STDOUT,check=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=7) as pool:
        for _ in pool.map(run,range(7)):pass
    print('All seven calibrated overlays complete',flush=True)


if __name__=='__main__':main()
