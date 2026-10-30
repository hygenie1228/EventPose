"""Read-only Track 2 RGB/MHR inputs and independently cached SMPL-X supervision.

Topology maps are the official Meta MHR maps. Parameter fitting is approximate;
all fit residuals are retained. GT is decoded in OpenCV world coordinates,
then transformed with the existing world-to-camera calibration for supervision.
"""
from pathlib import Path
import hashlib
import json
import random
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'exp2/hj/challenge_track2_v6'
OUTPUT = ROOT / 'exp/261025/track2_multihmr_12h'
ASSETS = ROOT / 'data/human_model_files'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.partial.json')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def open_track2_target_mask(sample):
    """Open unchanged mask bytes with bounded decoder threads; never write source."""
    import cv2
    source = SOURCE / sample / 'processed/human_mask.mp4'
    index = OUTPUT / 'work/target_mask_video_cache.json'
    path = source
    if index.exists():
        item = json.loads(index.read_text())['samples'][sample]
        path = Path(item['cache_path'])
        assert sha256(source) == item['source_sha256'] == sha256(path), sample
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f'Cannot decode target mask for {sample}: {path}')
    return cap


def plan():
    """Freeze sequence-level 20/10 split, with 4 Train-only validation sequences."""
    destination = OUTPUT / 'manifest.json'
    if destination.exists():
        value = json.loads(destination.read_text())
        assert len(value['train']) == 20 and len(value['test']) == 10
        for sample, record in value['samples'].items():
            assert sha256(SOURCE / sample / 'human/human_params_gt.npz') == record['gt_sha256']
            if 'world_to_camera' not in record:
                meta = json.loads((SOURCE / sample / 'meta_info.json').read_text())
                record['world_to_camera'] = meta['camera_extrinsics']
        save_json(destination, value)
        return value
    samples = sorted(p.parent.parent.name for p in SOURCE.glob('*/human/human_params_gt.npz'))
    assert len(samples) == 30, len(samples)
    random.Random(261025).shuffle(samples)
    train, test = sorted(samples[:20]), sorted(samples[20:])
    validation = sorted(random.Random(261026).sample(train, 4))
    fitting = sorted(set(train) - set(validation))
    records = {}
    for name in sorted(samples):
        source = SOURCE / name
        meta = json.loads((source / 'meta_info.json').read_text())
        intrinsic = json.loads((source / 'processed/intrinsic.json').read_text())
        K = [[intrinsic['fx'], 0, intrinsic['cx']],
             [0, intrinsic['fy'], intrinsic['cy']], [0, 0, 1]]
        with np.load(source / 'human/human_params_gt.npz') as z:
            assert z['pose'].shape == (meta['num_frames'], 136)
            assert z['shape'].shape == (45,) and z['scales'].shape == (68,)
            assert all(np.isfinite(z[k]).all() for k in ['pose', 'shape', 'scales'])
        records[name] = dict(frames=meta['num_frames'], fps=meta['fps'],
            width=meta['width'], height=meta['height'], K=K,
            world_to_camera=meta['camera_extrinsics'],
            gt_sha256=sha256(source / 'human/human_params_gt.npz'),
            metadata_sha256=sha256(source / 'meta_info.json'),
            intrinsic_sha256=sha256(source / 'processed/intrinsic.json'),
            camera_is_released_gt=meta.get('camera_is_released_gt'),
            camera_provenance=meta.get('camera_provenance'))
    value = dict(seed=261025, train=train, test=test, training=fitting,
        validation=validation, samples=records, created_unix=time.time(),
        training_fps=2, validation_frames_per_sequence=64,
        selection='Train-only sequence-disjoint validation; Test never selects checkpoints',
        final_export='Each of all 30 source sequences: human/human_params_uranus.npz',
        source_read_only_except_final_uranus_export=True,
        camera_caveat='Existing K calibration includes GT-assisted estimation; this is a calibrated local experiment, not a blind camera benchmark.',
        identity_caveat='Sequence-disjoint, not asserted person-disjoint: source metadata does not identify actors.')
    assert not set(train) & set(test)
    save_json(destination, value)
    return value


def decode_mhr(model, pose, shape, scales):
    """Exact reference-frame decoder, with gradients enabled for reverse fitting."""
    import torch
    n = len(pose)
    shape = shape.reshape(-1, 45).expand(n, -1).contiguous()
    scales = scales.reshape(-1, 68).expand(n, -1)
    v, j = model(shape, torch.cat([pose, scales], 1),
                 torch.zeros(n, 72, device=pose.device, dtype=pose.dtype), True)
    flip = torch.tensor([1., -1., -1.], device=pose.device)
    return v * flip / 100., j[..., :3] * flip / 100.


def mapping(name, faces, device):
    import torch
    with np.load(ASSETS / 'mhr/conversion' / (name + '_mapping.npz')) as z:
        ids = torch.as_tensor(np.array(faces)[z['triangle_ids']], device=device, dtype=torch.long)
        weights = torch.as_tensor(z['baryc_coords'], device=device, dtype=torch.float32)
    assert torch.allclose(weights.sum(1), torch.ones(len(weights), device=device), atol=1e-5)
    return ids, weights


def transfer(vertices, ids, weights):
    return (vertices[:, ids] * weights[None, :, :, None]).sum(2)


def fit_smplx(body, target, iterations=350):
    """Fit mapped GT mesh in metres, full pose and a batch-shared shape.

    Only offline label generation calls this function. Initialization uses mesh
    pelvis/hips/spine; no RGB estimator or Test prediction participates.
    """
    import torch
    import roma
    n = len(target)
    device = target.device
    reg = body.J_regressor
    joints = reg @ target
    zeros = lambda width: torch.zeros(n, width, device=device)
    extra = dict(expression=zeros(10), leye_pose=zeros(3), reye_pose=zeros(3))
    with torch.no_grad():
        rest = body(betas=zeros(10), global_orient=zeros(3), body_pose=zeros(63),
                    left_hand_pose=zeros(45), right_hand_pose=zeros(45),
                    jaw_pose=zeros(3), transl=zeros(3), **extra).vertices
        rest_joints = reg @ rest
        # Three torso vectors determine a proper root orientation independently
        # for each frame, avoiding arbitrary camera-axis initializations.
        x = rest_joints[:, [1, 2, 3]] - rest_joints[:, :1]
        y = joints[:, [1, 2, 3]] - joints[:, :1]
        u, _, vh = torch.linalg.svd(x.transpose(1, 2) @ y)
        d = torch.ones(n, 3, device=device)
        d[:, -1] = torch.det(u @ vh).sign()
        rotation = ((u * d[:, None]) @ vh).transpose(1, 2)
        root_init = roma.rotmat_to_rotvec(rotation)
    root = root_init.detach().clone().requires_grad_()
    pose = torch.zeros(n, 63, device=device, requires_grad=True)
    hands = torch.zeros(n, 90, device=device, requires_grad=True)
    jaw = torch.zeros(n, 3, device=device, requires_grad=True)
    beta = torch.zeros(1, 10, device=device, requires_grad=True)
    trans = (joints[:, 0] - rest_joints[:, 0]).detach().clone().requires_grad_()
    params = [root, pose, hands, jaw, beta, trans]
    optimizer = torch.optim.Adam(params, lr=.03)
    def forward():
        return body(betas=beta.expand(n, -1), global_orient=root,
                    body_pose=pose, left_hand_pose=hands[:, :45],
                    right_hand_pose=hands[:, 45:], jaw_pose=jaw, transl=trans, **extra).vertices
    for step in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        vertices = forward()
        vertex_loss = (vertices - target).square().mean()
        joint_loss = ((reg @ vertices)[:, :22] - joints[:, :22]).square().mean()
        loss = vertex_loss + .1 * joint_loss + 1e-7 * beta.square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        for group in optimizer.param_groups:
            group['lr'] = .03 if step < iterations * .5 else .008 if step < iterations * .85 else .002
        torch.nn.utils.clip_grad_norm_(params, 5.)
        optimizer.step()
    with torch.no_grad():
        vertices = forward()
        result_joints = reg @ vertices
        rotations = roma.rotvec_to_rotmat(torch.cat([
            root[:, None], pose.reshape(n, 21, 3), hands[:, :45].reshape(n, 15, 3),
            hands[:, 45:].reshape(n, 15, 3), jaw[:, None]], 1))
        return dict(vertices=vertices.cpu().numpy(), joints=result_joints.cpu().numpy(),
            rotations=rotations.cpu().numpy(), betas=beta.expand(n, -1).cpu().numpy(),
            global_orient=root.cpu().numpy(), body_pose=pose.cpu().numpy(),
            left_hand_pose=hands[:, :45].cpu().numpy(), right_hand_pose=hands[:, 45:].cpu().numpy(),
            jaw_pose=jaw.cpu().numpy(), transl=trans.cpu().numpy(),
            fit_error_mm=(vertices - target).norm(dim=-1).mean(1).mul(1000).cpu().numpy())


def prepare_sample(sample, iterations=350, batch_size=24, limit=0):
    """Convert only Train/validation labels, cache sampled RGB outside source."""
    import cv2
    import smplx
    import torch
    from utils.pose_evaluation import configure_torch
    configure_torch()
    manifest = json.loads((OUTPUT / 'manifest.json').read_text())
    assert sample in manifest['train'], 'Never prepare Test labels for training'
    record = manifest['samples'][sample]
    role = 'training' if sample in manifest['training'] else 'validation'
    indices = (np.arange(0, record['frames'], max(1, round(record['fps'] / 2)), dtype=int)
        if role == 'training' else np.unique(np.linspace(0, record['frames'] - 1,
            min(64, record['frames']), dtype=int)))
    if limit:
        indices = indices[:limit]
    output = OUTPUT / ('smoke_camera' if limit else 'work')
    destination = output / 'targets' / (sample + '.npz')
    if destination.exists():
        with np.load(destination) as z:
            assert z['source_sha256'].item() == record['gt_sha256']
            assert np.array_equal(z['frames'], indices)
        print('Verified existing', destination, flush=True)
        return
    mhr = torch.jit.load(str(ASSETS / 'mhr/mhr_model.pt'), map_location='cuda').eval()
    for p in mhr.parameters():
        p.requires_grad_(False)
    faces = mhr.state_dict()['character_torch.mesh.faces'].cpu().numpy()
    ids, weights = mapping('mhr2smplx', faces, 'cuda')
    body = smplx.create(str(ASSETS), model_type='smplx', gender='neutral',
        use_pca=False, flat_hand_mean=True, num_betas=10, batch_size=1).cuda().eval()
    for p in body.parameters():
        p.requires_grad_(False)
    with np.load(SOURCE / sample / 'human/human_params_gt.npz') as z:
        p = torch.as_tensor(z['pose'][indices], device='cuda', dtype=torch.float32)
        b = torch.as_tensor(z['shape'], device='cuda', dtype=torch.float32)
        s = torch.as_tensor(z['scales'], device='cuda', dtype=torch.float32)
    parts = []
    started = time.time()
    for start in range(0, len(indices), batch_size):
        with torch.no_grad():
            v, _ = decode_mhr(mhr, p[start:start + batch_size], b, s)
            extrinsic = torch.as_tensor(record['world_to_camera'], device='cuda', dtype=torch.float32)
            v = v @ extrinsic[:3, :3].T + extrinsic[:3, 3]
            target = transfer(v, ids, weights)
        result = fit_smplx(body, target, iterations)
        parts.append(result)
        print(json.dumps(dict(sample=sample, converted=min(start + batch_size, len(indices)),
            total=len(indices), mean_fit_mm=float(result['fit_error_mm'].mean()),
            elapsed_seconds=time.time() - started)), flush=True)
    result = {k:np.concatenate([x[k] for x in parts]) for k in parts[0]}
    assert all(np.isfinite(x).all() for x in result.values())
    image_folder = output / 'images' / sample
    image_folder.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(SOURCE / sample / 'video.mp4'))
    assert cap.isOpened()
    selected = set(indices.tolist())
    written = 0
    for frame in range(indices[-1] + 1):
        ok, image = cap.read()
        assert ok, (sample, frame)
        if frame in selected:
            assert image.shape[:2] == (record['height'], record['width'])
            assert cv2.imwrite(str(image_folder / f'{frame:06d}.jpg'), image,
                [cv2.IMWRITE_JPEG_QUALITY, 97])
            written += 1
    cap.release()
    assert written == len(indices)
    joints = result['joints'][:, None].astype(np.float32)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix('.partial.npz')
    np.savez_compressed(tmp, joints=joints, centers=joints[:, :, 15],
        rotations=result['rotations'][:, None], betas=result['betas'][:, None],
        valid=np.ones((len(indices), 1), bool), frames=indices,
        K=np.array(record['K'], np.float32), image_shape=[record['height'], record['width']],
        source_sha256=np.asarray(record['gt_sha256']), fit_error_mm=result['fit_error_mm'],
        **{k:v for k,v in result.items() if k not in ['vertices', 'joints', 'rotations', 'betas', 'fit_error_mm']})
    tmp.replace(destination)
    save_json(output / 'receipts' / (sample + '.json'), dict(sample=sample, role=role,
        frames=len(indices), iterations=iterations, mean_fit_mm=float(result['fit_error_mm'].mean()),
        max_frame_fit_mm=float(result['fit_error_mm'].max()),
        elapsed_seconds=time.time() - started, source_sha256=record['gt_sha256'],
        method='Official Meta barycentric map + full-surface SMPL-X Adam fitting',
        shape_policy='shared within fitting batches; original MHR identity remains shared',
        coordinate_frame='OpenCV camera, world-to-camera calibration applied exactly once'))


def finalize_indices():
    manifest = json.loads((OUTPUT / 'manifest.json').read_text())
    work = OUTPUT / 'work'
    for role in ['training', 'validation']:
        records = []
        for sample in manifest[role]:
            with np.load(work / 'targets' / (sample + '.npz')) as z:
                assert z['source_sha256'].item() == manifest['samples'][sample]['gt_sha256']
                for row, frame in enumerate(z['frames']):
                    assert (work / 'images' / sample / f'{frame:06d}.jpg').is_file()
                    records.append(dict(session=sample, row=row, frame=int(frame)))
        save_json(work / (role + '_index.json'), records)
    import smplx
    body = smplx.create(str(ASSETS), model_type='smplx', gender='neutral',
        use_pca=False, flat_hand_mean=True, num_betas=10)
    np.save(work / 'joint_regressor.npy', body.J_regressor.numpy())
    save_json(work / 'complete.json', dict(training_sequences=16, validation_sequences=4,
        test_sequences_untouched=10, finished_unix=time.time()))
