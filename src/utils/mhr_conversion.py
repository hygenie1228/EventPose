"""Approximate prediction-only SMPL-X surface to shared-identity MHR fitting.

Uses Meta topology correspondence and the unchanged V2D reference fitter on
three anchor frames. Remaining frame poses use differentiable surface fitting;
no temporal smoothing, GT pose, GT identity, or GT locations are used.
"""
import json
import time
import numpy as np
import torch

from multihmr.track2_data import ASSETS, decode_mhr, mapping, transfer
from utils import mhr_reference_fitting as reference


def fit_prediction_sequence(smpl_vertices_cam, smpl_faces, world_to_camera,
                            output_path, deadline, batch_size=32, iterations=160):
    n = len(smpl_vertices_cam)
    assert n >= 3 and np.isfinite(smpl_vertices_cam).all()
    model_path = ASSETS / 'mhr/mhr_model.pt'
    model = torch.jit.load(str(model_path), map_location='cuda').eval()
    for p in model.parameters():
        p.requires_grad_(False)
    ids, weights = mapping('smplx2mhr', smpl_faces, 'cuda')
    targets = np.empty((n, 18439, 3), np.float32)
    with torch.no_grad():
        for start in range(0, n, 32):
            v = torch.as_tensor(smpl_vertices_cam[start:start + 32], device='cuda')
            # Preserve the RGB camera frame, matching existing native MHR
            # predictions/renderers. The official scorer handles the fixed
            # world/camera difference with its frame-0 global Sim(3).
            targets[start:start + 32] = transfer(v, ids, weights).cpu().numpy()
    keys = np.unique(np.linspace(0, n - 1, min(3, n), dtype=int))
    started = time.time()
    initial = reference.convert(targets[keys], str(model_path), device='cuda:0',
        n_keyframes=3, precision='float32', chunk=1, model_batch=64,
        log=lambda *args, **kwargs: print(*args, **kwargs, flush=True))
    shape = torch.as_tensor(initial['shape'], device='cuda')
    scales = torch.as_tensor(initial['scales'], device='cuda')
    anchor_pose = initial['pose'].copy()
    anchor_pose[:, 3:6] = np.unwrap(anchor_pose[:, 3:6], axis=0)
    interpolated = np.stack([np.interp(np.arange(n), keys, anchor_pose[:, i])
                            for i in range(136)], 1).astype(np.float32)
    fitted = np.empty((n, 136), np.float32)
    errors = np.empty(n, np.float32)
    for start in range(0, n, batch_size):
        if time.time() >= deadline - 30:
            raise TimeoutError('12-hour deadline during MHR conversion')
        stop = min(n, start + batch_size)
        target = torch.as_tensor(targets[start:stop], device='cuda')
        pose = torch.tensor(interpolated[start:stop], device='cuda', requires_grad=True)
        with torch.no_grad():
            vertex, _ = decode_mhr(model, pose, shape, scales)
            # Native MHR root translation unit is 10 cm, with Y/Z flipped.
            pose[:, :3] += (target.mean(1) - vertex.mean(1)) * torch.tensor(
                [10., -10., -10.], device='cuda')
        optimizer = torch.optim.Adam([pose], lr=.025)
        best_pose = pose.detach().clone()
        best_error = torch.full((len(pose),), float('inf'), device='cuda')
        for step in range(iterations):
            optimizer.zero_grad(set_to_none=True)
            vertex, _ = decode_mhr(model, pose, shape, scales)
            squared = (vertex - target).square().mean((1, 2))
            with torch.no_grad():
                better = squared < best_error
                best_pose[better] = pose[better]
                best_error[better] = squared[better]
            loss = squared.mean()
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([pose], 1.)
            for group in optimizer.param_groups:
                group['lr'] = .025 if step < iterations * .6 else .005
            optimizer.step()
        with torch.no_grad():
            vertex, _ = decode_mhr(model, best_pose, shape, scales)
            errors[start:stop] = (vertex - target).norm(dim=-1).mean(1).mul(1000).cpu().numpy()
            fitted[start:stop] = best_pose.cpu().numpy()
        print(json.dumps(dict(event='mhr_pose_fit', frames=stop, total=n,
            mean_fit_mm=float(errors[start:stop].mean()), elapsed_seconds=time.time() - started)), flush=True)
    report = dict(conversion='Meta barycentric mapping; V2D reference 3-frame shared identity; differentiable independent per-frame pose refinement',
        shared_identity=True, temporal_smoothing=False, gt_parameters_used=False,
        coordinate_frame='opencv_camera decoded using reference axis flip',
        world_to_camera_used_for_fitting=False,
        iterations=iterations, anchor_frames=keys.tolist(), reference_anchor_report=initial['report'],
        mean_vertex_fit_mm=float(errors.mean()), p95_frame_fit_mm=float(np.percentile(errors, 95)),
        max_frame_fit_mm=float(errors.max()), elapsed_seconds=time.time() - started)
    assert np.isfinite(fitted).all() and np.isfinite(errors).all()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix('.partial.npz')
    np.savez_compressed(temporary, pose=fitted, scales=initial['scales'], shape=initial['shape'],
        per_frame_vertex_error_mm=errors, body_model=np.asarray('MHR'),
        report=np.asarray(json.dumps(report)))
    temporary.replace(output_path)
    return report


def refine_prediction_sequence(smpl_vertices_cam, smpl_faces, initial_path,
                               output_path, deadline, threshold_mm=20., iterations=10):
    """Version 2: fixed-identity, independent reference-LM pose polishing.

    Uses prediction-only residuals to select frames. Accepts a pose only when
    its full-mesh squared error decreases. No smoothing, GT, or metric inputs.
    The original conversion is immutable; results go to a separate NPZ.
    """
    from multihmr.track2_data import sha256
    started=time.time();initial_hash=sha256(initial_path)
    source_hash=sha256(initial_path.with_name('smplx_predictions.npz'))
    assert started<deadline,'Original 12-hour deadline reached'
    with np.load(initial_path) as z:
        fitted=z['pose'].copy();shape=z['shape'].copy();scales=z['scales'].copy()
        errors=z['per_frame_vertex_error_mm'].copy()
    baseline_errors=errors.copy();frames=np.flatnonzero(errors>threshold_mm)
    assert len(smpl_vertices_cam)==len(fitted) and np.isfinite(smpl_vertices_cam).all()
    assert np.isfinite(fitted).all() and np.isfinite(errors).all()
    wrapped=reference.MHR(str(ASSETS/'mhr/mhr_model.pt'),'cuda:0',chunk=256,precision='float32')
    ids,weights=mapping('smplx2mhr',smpl_faces,'cuda')
    identity=torch.tensor(np.r_[scales,shape],device='cuda',dtype=torch.float64)[None]
    progress=output_path.with_suffix('.progress.npz');offset=0;accepted=0
    if progress.exists():
        with np.load(progress) as z:
            assert z['initial_sha256'].item()==initial_hash and z['source_sha256'].item()==source_hash,'Refinement source changed'
            assert z['threshold_mm'].item()==threshold_mm and z['iterations'].item()==iterations
            fitted=z['pose'].copy();errors=z['errors'].copy();offset=int(z['offset']);accepted=int(z['accepted'])
    for start in range(offset,len(frames),32):
        if time.time()>=deadline-30:raise TimeoutError('12-hour deadline during revision2 MHR refinement')
        chosen=frames[start:start+32]
        with torch.no_grad():
            target=transfer(torch.tensor(smpl_vertices_cam[chosen],device='cuda'),ids,weights).double().reshape(len(chosen),-1)*1000
            before=torch.tensor(fitted[chosen],device='cuda',dtype=torch.float64)
            original=wrapped.verts(before,identity)
            original_cost=(original-target).square().mean(1)
            solved,_=reference.lm_pose(wrapped,target,before,identity,iters=iterations,frames_per_batch=8)
            # Recheck the actual float32 parameters written to disk.
            solved=solved.float().double();candidate=wrapped.verts(solved,identity)
            better=(candidate-target).square().mean(1)<original_cost
            assert torch.isfinite(candidate).all() and torch.isfinite(solved).all()
            final=torch.where(better[:,None],candidate,original)
            selected=torch.where(better[:,None],solved,before)
            fitted[chosen]=selected.cpu().numpy().astype(np.float32)
            errors[chosen]=(final-target).reshape(len(chosen),-1,3).norm(dim=-1).mean(1).cpu().numpy()
            accepted+=int(better.sum())
        partial=progress.with_name(progress.stem+'.partial.npz')
        np.savez_compressed(partial,pose=fitted,errors=errors,offset=start+len(chosen),accepted=accepted,
            initial_sha256=np.asarray(initial_hash),source_sha256=np.asarray(source_hash),threshold_mm=threshold_mm,iterations=iterations)
        partial.replace(progress)
        print(json.dumps(dict(event='mhr_revision2',frames_refined=start+len(chosen),total_selected=len(frames),
            selected_batch_mean_fit_mm=float(errors[chosen].mean()),elapsed_seconds=time.time()-started)),flush=True)
    report=dict(conversion_revision=2,conversion='Version1 prediction followed by reference LM10 pose polishing of frames with mean residual above20mm',
        gt_parameters_used=False,temporal_smoothing=False,shared_identity=True,identity_refitted=False,
        frame_selection='prediction-only whole-mesh mean fitting residual',threshold_mm=threshold_mm,
        iterations=iterations,solver='unchanged reference lm_pose, float64 solver / float32 MHR, model batch256, frame batch8',
        acceptance='strictly lower full-mesh MSE after float32 pose encoding',initial_sha256=initial_hash,source_sha256=source_hash,
        selected_frames=len(frames),accepted_frames=accepted,total_frames=len(fitted),
        baseline_mean_vertex_fit_mm=float(baseline_errors.mean()),mean_vertex_fit_mm=float(errors.mean()),
        p95_frame_fit_mm=float(np.percentile(errors,95)),max_frame_fit_mm=float(errors.max()),
        coordinate_frame='opencv_camera decoded using reference axis flip',world_to_camera_used_for_fitting=False,
        elapsed_seconds=time.time()-started)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    temporary=output_path.with_suffix('.partial.npz')
    np.savez_compressed(temporary,pose=fitted,shape=shape,scales=scales,per_frame_vertex_error_mm=errors,
        body_model=np.asarray('MHR'),report=np.asarray(json.dumps(report)))
    temporary.replace(output_path)
    return report
