"""Frozen same-device RGB/event Test evaluation and common-population summaries.

This helper does not select methods or checkpoints using Test observations.
Call it through the project evaluation launcher, in a separate CUDA worker.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from utils.part_aggregation_metrics import (
    METRICS, common_detection_mask, score_chunk, summarize_chunks,
)

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+"."+str(os.getpid())+".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n")
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for b in iter(lambda: stream.read(8*1024**2), b""):
            h.update(b)
    return h.hexdigest()


def _tasks(config):
    out = Path(config["output_root"])
    source = Path(config.get("source_cache_root", ROOT/"exp/261028/device3_encoding_12h"))
    path = out/"test/tasks.json"
    if not path.exists():
        path = source/"test/tasks.json"
    tasks = read(path)
    if any(t.get("smoke") for t in tasks):
        raise ValueError("Full Test manifest must not contain smoke tasks")
    return tasks


def infer(model, collector, source, config):
    """Batch inference with GT confined to the downstream matching collector."""
    import torch
    collector.matching_sensor_geometry = (
        source.R, source.t, source.f["rgb3_K_native"][:].astype(np.float32),
        tuple(map(int, source.f["rgb3_native_shape"][:])),
    )
    collector.task["matching_protocol"] = "primary_sensor_native_2D_projection_then_common_camera_3D_errors_v1"
    pending = []; image_ids = []
    original = model.base.detection
    input_keys = tuple(model.input_keys)
    deadline = float(config["deadline_unix"])
    batch_size = int(config.get("evaluation_batch_size", config.get("batch_size", 8)))

    def detection(*args, **kwargs):
        result = original(*args, **kwargs)
        image_ids[:] = result[2][0].detach().cpu().tolist()
        return result

    def flush():
        if time.time() >= deadline-30:
            raise TimeoutError("Part aggregation evaluation reached its fixed deadline")
        selected = [(i, value) for i, (_, value) in enumerate(pending) if value is not None]
        grouped = [[] for _ in pending]
        if selected:
            batch = {k: torch.stack([value[k] for _, value in selected]).cuda(non_blocking=True)
                     for k in input_keys if k in selected[0][1]}
            control = config.get("evaluation_control", "intact")
            event_key = "event_bins"
            if event_key in batch:
                if control in ("zero_event", "missing_event"):
                    batch[event_key] = torch.zeros_like(batch[event_key])
                    if control == "missing_event":
                        for key in ("event_bin_available", "auxiliary_available"):
                            if key in batch:
                                batch[key] = torch.zeros_like(batch[key])
                elif control == "shuffle_event":
                    # Deterministic batch roll, not a GT-derived sample selection.
                    batch[event_key] = batch[event_key].roll(1, 0)
                elif control == "reverse_time":
                    batch[event_key] = batch[event_key].flip(1)
                elif control != "intact":
                    raise ValueError("Unknown event intervention: "+control)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                people = model(**batch, is_training=False, det_thresh=.3, nms_kernel_size=3)
            if len(people) != len(image_ids):
                raise ValueError("Detection identities do not match returned people")
            with torch.no_grad(), torch.autocast("cuda", enabled=False):
                for image_id, person in zip(image_ids, people):
                    joints = (collector.reg@person["v3d"].float()).cpu().numpy()
                    joints = (joints-source.t)@source.R
                    if not np.isfinite(joints).all():
                        raise ValueError("Non-finite predicted joints")
                    grouped[selected[image_id][0]].append(joints)
        for (position, _), joints in zip(pending, grouped):
            collector.current_pos = position
            collector.collect(np.stack(joints) if joints else np.empty((0, 55, 3), np.float32), collector.K)
        pending.clear()

    model.base.detection = detection
    try:
        for position in range(collector.task["video_frames"]):
            pending.append((position, source.sample(collector.task["start_frame"]+position)))
            if len(pending) >= batch_size:
                flush()
        if pending:
            flush()
    finally:
        model.base.detection = original


def evaluate(config, rank=0, workers=1):
    """Evaluate one condition's frozen best checkpoint on a Test session shard."""
    import torch
    import signal
    import utils.pose_evaluation as base
    from .part_aggregation_data import MemorySensors
    from .part_aggregation_factory import build_part_model
    if isinstance(config, (str, Path)):
        config = read(config)
    out = Path(config["output_root"]); result_root = out/"results"
    result_root.mkdir(parents=True, exist_ok=True)
    if sha(ROOT/"dataset/split.json") != config["split_sha256"]:
        raise ValueError("Dataset split changed after the training protocol was frozen")
    if int(os.environ.get("EVENTPOSE_PHYSICAL_GPU", "-1")) not in config["allowed_gpus"]:
        raise ValueError("Evaluation worker is not on an authorized physical GPU")
    torch.set_num_threads(2); torch.cuda.set_device(0)
    def interrupted(*_):raise TimeoutError('Evaluation interrupted; preserve partial collector')
    signal.signal(signal.SIGTERM,interrupted)
    source_root = Path(config.get("source_cache_root", ROOT/"exp/261028/device3_encoding_12h"))
    regressor = out/"joint_regressor.npy"
    if not regressor.exists():
        regressor = source_root/"joint_regressor.npy"
    destination = result_root/"joint_regressor.npy"
    if not destination.exists():
        temporary = destination.with_name(destination.name+"."+str(os.getpid())+".tmp")
        temporary.write_bytes(regressor.read_bytes()); temporary.replace(destination)
    base.OUT = result_root
    checkpoint = Path(config.get("evaluation_checkpoint", out/"checkpoints"/config["name"]/"best.pt"))
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved.get("test_used_for_selection") is not False or saved.get("successful_steps", 0) <= 0:
        raise ValueError("Evaluation requires an actual Train-selected checkpoint")
    # Checkpoint remains immutable for the entire evaluation worker.
    checkpoint_sha = sha(checkpoint)
    model = build_part_model(config, device="cuda", checkpoint=checkpoint).eval()
    tasks = _tasks(config)
    names = sorted({t["parent_session"] for t in tasks})[int(rank)::int(workers)]
    label = config.get("evaluation_name", config["name"])
    provenance = dict(pid=os.getpid(), checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha,
        gpu_uuid=os.environ.get("CUDA_VISIBLE_DEVICES"), physical_gpu=os.environ.get("EVENTPOSE_PHYSICAL_GPU"),
        cuda_name=torch.cuda.get_device_name(), rank=rank, workers=workers, parent_sessions=names,
        test_used_for_selection=False, GT_used_as_model_input=False,
        evaluation_control=config.get("evaluation_control", "intact"), started_unix=time.time())
    write(result_root/label/("execution_rank"+str(rank)+".json"), provenance)
    for name in names:
        clips = [t for t in tasks if t["parent_session"] == name]
        if all((result_root/label/t["session"]/"complete.json").exists() for t in clips):
            for task in clips:
                previous = read(result_root/label/task["session"]/"complete.json")["task"]
                if previous.get("checkpoint_sha256") != checkpoint_sha:
                    raise ValueError("Completed Test session uses another checkpoint")
            continue
        source = MemorySensors(name, source_root=source_root, cache_root=out, use_events=config['variant']!='rgb_only')
        try:
            for raw in clips:
                if time.time() >= config["deadline_unix"]-60:
                    raise TimeoutError("Full Test evaluation stopped at its fixed deadline")
                folder = result_root/label/raw["session"]
                if (folder/"complete.json").exists():
                    previous = read(folder/"complete.json")["task"]
                    if previous.get("checkpoint_sha256") != checkpoint_sha:
                        raise ValueError("Existing Test result uses another checkpoint")
                    continue
                task = dict(raw, checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha,
                    condition="rgb3_event3", split_sha256=config["split_sha256"],
                    aggregation=config.get("aggregation"), evaluation_control=config.get("evaluation_control", "intact"))
                collector = base.Collector(task, label)
                try:
                    infer(model, collector, source, config); collector.save()
                except Exception:
                    collector.save(partial=True)
                    raise
        finally:
            source.close()
    if sha(checkpoint) != checkpoint_sha:
        raise ValueError("Checkpoint was modified during evaluation")
    provenance.update(completed_unix=time.time(), completed=True)
    write(result_root/label/("execution_rank"+str(rank)+".json"), provenance)
    return provenance


def analyze(config, names=None):
    """Summarize completed conditions on full Test; partial Test is never a mean."""
    from utils.paired_statistics import paired_cluster, holm_adjust
    if isinstance(config, (str, Path)):
        config = read(config)
    out = Path(config["output_root"]); tasks = _tasks(config)
    if names is None:
        names = config.get("comparison_names", ["rgb_only", "fixed", "global_dynamic", "part_dynamic"])
    complete = [n for n in names if all((out/"results"/n/t["session"]/"complete.json").exists() for t in tasks)]
    pending = [n for n in names if n not in complete]
    groups = {"all_completed": complete}
    reference = config.get("reference_name", "rgb_only")
    if reference in complete:
        groups.update({"pair_"+n: [reference, n] for n in complete if n != reference})
    groups.update({"own_"+n: [n] for n in complete})
    rows = {g: {n: [] for n in members} for g, members in groups.items()}
    coverage = dict.fromkeys(complete, 0); gt_total = 0; result_hashes = []
    for task in tasks:
        with np.load(task["gt_file"]) as z:
            gt = z["joints_cam_m"]; valid = z["valid"]; positions = z["source_rgb_positions"]
            if positions[0] != 0 or len(positions) != task["video_frames"]:
                raise ValueError("Test segment does not start at its exact original frame0")
        gt_total += int(valid.sum()); predictions = {}
        for name in complete:
            folder = out/"results"/name/task["session"]
            receipt = read(folder/"complete.json")
            if receipt.get("partial") or receipt["task"].get("smoke"):
                raise ValueError("Partial/smoke output cannot be reported as full Test")
            if receipt["task"].get("split_sha256") != config["split_sha256"]:
                raise ValueError("Completed Test task uses another data split")
            for key in ("parent_session", "start_frame", "end_frame", "video_frames"):
                if receipt["task"][key] != task[key]:
                    raise ValueError("Task source interval changed: "+key)
            with np.load(folder/"result.npz") as z:
                if not z["seen_gt_rows"].all() or not np.array_equal(z["source_rgb_positions"], positions):
                    raise ValueError("Test task contains unvisited or misaligned GT rows")
                pred = z["matched_joints_cam_m"]
                mask = valid & (z["matched_pred_indices"] >= 0)
                if pred.shape != gt.shape or not np.isfinite(pred[mask]).all():
                    raise ValueError("Invalid matched prediction shape or value")
                if np.isfinite(pred[valid & ~mask]).any():
                    raise ValueError("An unmatched valid GT has fabricated prediction coordinates")
                predictions[name] = pred
                coverage[name] += int(mask.sum())
            result_hashes.append(dict(condition=name, segment=task["session"], sha256=sha(folder/"result.npz")))
        for group, members in groups.items():
            if not members:
                continue
            common = common_detection_mask({n: predictions[n] for n in members}, gt, valid)
            for name in members:
                row, _, _ = score_chunk(predictions[name], gt, valid, positions, common=common)
                row.update(session=task["session"], parent_session=task["parent_session"],
                           cluster=int(task["parent_session"].split("_")[3]))
                rows[group][name].append(row)
    populations = {g: {n: summarize_chunks(rr) for n, rr in members.items()} for g, members in rows.items()}
    tests = {}
    for name in complete:
        if name == reference or reference not in complete:
            continue
        group = rows["pair_"+name]
        clusters = sorted({r["cluster"] for r in group[reference]})
        for metric, index in [("frame_pa", 3), ("RPE_Z", 4)]:
            sums = {}; counts = None
            for n in (reference, name):
                selected = [[r for r in group[n] if r["cluster"] == c] for c in clusters]
                sums[n] = [1000*sum(r["sums_m"][index] for r in values) for values in selected]
                current = [sum(r["count"] for r in values) for values in selected]
                if counts is not None and current != counts:
                    raise ValueError("Paired statistics use different populations")
                counts = current
            tests[name+"_"+metric] = paired_cluster(sums[reference], sums[name], counts, seed=261029)
    for suffix in ("_frame_pa", "_RPE_Z"):
        for name, value in holm_adjust({k: v["p_value"] for k, v in tests.items() if k.endswith(suffix)}).items():
            tests[name]["holm_p_value"] = value
    result = dict(updated_unix=time.time(), conditions=list(names), completed=complete, pending=pending,
                  all_complete=not pending, segments=len(tasks), frames=sum(t["video_frames"] for t in tasks),
                  GT_instances=gt_total, metrics=populations.get("all_completed", {}), populations=populations,
                  coverage={n: dict(matched=coverage[n], total=gt_total, fraction=coverage[n]/gt_total) for n in complete},
                  tests=tests, Test_used_for_selection=False, metrics_keys=METRICS,
                  root_depth_definition="mean(abs(pred pelvis Z - GT pelvis Z))*1000 before alignment",
                  population="same matched GT instances; exact frame0 required in each person/chunk")
    write(out/"comparison.json", result)
    write(out/"segment_sums.json", rows)
    write(out/"raw_prediction_hashes.json", result_hashes)
    return result


def report_table(result):
    """Korean report table; unmeasured rows retain em dashes."""
    labels = dict(rgb_only="RGB 단독", fixed="고정 3구간 균등 집계", global_dynamic="전신 동적 집계",
                  part_dynamic="부위별 동적 집계")
    lines = ["| 방법 | 첫 프레임 MPJPE ↓ | 첫 프레임 PA-MPJPE ↓ | 프레임별 MPJPE ↓ | 프레임별 PA-MPJPE ↓ | RPE-Z ↓ |",
             "|---|---:|---:|---:|---:|---:|"]
    measured = result.get("metrics", {})
    for name in result["conditions"]:
        row = measured.get(name, {})
        values = []
        for key in METRICS:
            value = row.get(key)
            cell = "—" if value is None else f"{value:.2f}"
            if result.get("all_complete") and value is not None:
                available = [r[key] for r in measured.values() if r.get(key) is not None]
                if available and np.isclose(value, min(available), atol=.005, rtol=0):
                    cell = "**"+cell+"**"
            values.append(cell)
        lines.append("| "+labels.get(name, name)+" | "+" | ".join(values)+" |")
    lines += ["", "단위는 mm. 완료 조건에 공통 검출된 같은 GT 사람·프레임에서 계산한다. "
              "첫 프레임 정렬은 각 10초 구간의 정확한 frame0에서 구한 변환을 고정한다. "
              "RPE-Z는 정렬 전 pelvis의 Z축 절대 오차이며 3차원 거리와 다르다."]
    return "\n".join(lines)
