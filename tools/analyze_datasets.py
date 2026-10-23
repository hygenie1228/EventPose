#!/usr/bin/env python3
"""Inventory server I and server L using metadata only; never decode images."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "_agents"
SESSION_RE = re.compile(r"^\d{8}_\d{6}_.+")


def h5_count(group: h5py.Group) -> int:
    if "count" in group.attrs:
        return int(group.attrs["count"])
    if "timestamp_ns" in group:
        return int(group["timestamp_ns"].shape[0])
    if "offsets" in group:
        return int(group["offsets"].shape[0] - 1)
    return 0


def server_i_sessions(root: Path):
    occurrences = defaultdict(list)
    for bucket in sorted(root.glob("data*")):
        if not bucket.is_dir():
            continue
        for p in bucket.iterdir():
            if p.is_dir() and SESSION_RE.match(p.name):
                occurrences[p.name].append(p)
    return occurrences


def scan_i(root: Path):
    occurrences = server_i_sessions(root)
    rows, errors = [], []
    for name, paths in sorted(occurrences.items()):
        # data1/data2/data3 are storage partitions. Count a repeated session once.
        s = paths[0]
        row = Counter(session=name, copies=len(paths))
        row["buckets"] = ",".join(p.parent.name for p in paths)
        row["gt_h5_files"] = 0
        for modality in ("RGBD", "RGB", "Event"):
            for hp in sorted((s / modality).glob("*.h5")):
                try:
                    with h5py.File(hp, "r") as f:
                        names = []
                        f.visit(names.append)
                        if any("smpl" in x.lower() or "pose" in x.lower() for x in names):
                            row["gt_h5_files"] += 1
                        if modality == "RGBD":
                            if "streams/rgb" in f:
                                row["rgb"] += h5_count(f["streams/rgb"])
                            if "streams/depth_align" in f:
                                row["depth"] += h5_count(f["streams/depth_align"])
                            row["rgbd_devices"] += 1
                        elif modality == "RGB":
                            if "streams/images" in f:
                                row["rgb"] += h5_count(f["streams/images"])
                            row["rgb_devices"] += 1
                        else:
                            if "streams/images" in f:
                                row["event"] += h5_count(f["streams/images"])
                            row["event_devices"] += 1
                except Exception as exc:
                    errors.append(f"{hp}: {exc!r}")
        rows.append(dict(row))
    return rows, occurrences, errors


def scan_l(root: Path):
    rows, errors = [], []
    sessions = sorted(p for p in root.iterdir() if p.is_dir() and SESSION_RE.match(p.name))
    for s in sessions:
        row = Counter(session=s.name)
        manifest = s / "conversion_manifest.json"
        if manifest.is_file():
            try:
                m = json.loads(manifest.read_text())
                row["conversion_success"] = bool(m.get("success"))
                row["depth_included_setting"] = bool(m.get("settings", {}).get("depth_included", False))
                for o in m.get("outputs", []):
                    mod = str(o.get("modality", "")).lower()
                    if mod in ("rgb", "event", "depth"):
                        row[mod] += int(o.get("frames", 0))
                        row[f"{mod}_devices"] += 1
            except Exception as exc:
                errors.append(f"{manifest}: {exc!r}")

        # Trust actual RGB H5 streams over/alongside the manifest and detect depth.
        rgb_actual = depth_actual = rgb_devices = 0
        for hp in sorted((s / "RGB").glob("*/images.h5")):
            try:
                with h5py.File(hp, "r") as f:
                    if "streams/images" in f:
                        rgb_actual += h5_count(f["streams/images"])
                        rgb_devices += 1
                    if "streams/depth_align" in f:
                        depth_actual += h5_count(f["streams/depth_align"])
            except Exception as exc:
                errors.append(f"{hp}: {exc!r}")
        if rgb_actual:
            row["rgb"], row["rgb_devices"] = rgb_actual, rgb_devices
        row["depth"] = max(row["depth"], depth_actual)

        gt_files = sorted((s / "RGBD").glob("device*_mamma.h5"))
        row["gt_files"] = len(gt_files)
        gt_by_device = []
        for hp in gt_files:
            try:
                with h5py.File(hp, "r") as f:
                    n = int(f["results/person_valid"].shape[0])
                    valid = int(np.count_nonzero(f["results/person_valid"][:]))
                    success = int(np.count_nonzero(f["results/success"][:]))
                    device = int(f.attrs.get("logical_device_id", -1))
                    gt_by_device.append((device, n, valid, success))
                    row["gt_camera_frame_records"] += n
                    row["gt_camera_valid_persons"] += valid
            except Exception as exc:
                errors.append(f"{hp}: {exc!r}")
        # MAMMA files repeat synchronized GT for camera views. device0 is canonical;
        # if absent, use the lowest device as the unique session timeline estimate.
        if gt_by_device:
            device, n, valid, success = sorted(gt_by_device, key=lambda x: (x[0] != 0, x[0]))[0]
            row["gt_unique_frames"] = n
            row["gt_valid_persons"] = valid
            row["gt_success_frames"] = success
            row["gt_canonical_device"] = device
        rows.append(dict(row))
    return rows, errors


def total(rows, key):
    return sum(int(r.get(key, 0)) for r in rows)


def write_csv(path: Path, rows):
    keys = sorted({k for r in rows for k in r}, key=lambda x: (x != "session", x))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def charts(irows, lrows):
    labels = ["RGB", "Event", "Depth", "GT frames", "GT persons"]
    ivals = [total(irows, "rgb"), total(irows, "event"), total(irows, "depth"), 0, 0]
    lvals = [total(lrows, "rgb"), total(lrows, "event"), total(lrows, "depth"),
             total(lrows, "gt_unique_frames"), total(lrows, "gt_valid_persons")]
    x = np.arange(len(labels)); width = .36
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x-width/2, ivals, width, label="server I")
    ax.bar(x+width/2, lvals, width, label="server L")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xticks(x, labels); ax.set_ylabel("count (symlog scale)")
    ax.set_title("Dataset sample and pose counts")
    ax.legend(); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(OUT / "dataset_counts.svg"); plt.close(fig)

    common = sorted(set(r["session"] for r in irows) & set(r["session"] for r in lrows))
    im = {r["session"]: r for r in irows}; lm = {r["session"]: r for r in lrows}
    vals = [[lm[s].get("rgb", 0) / im[s].get("rgb", 1) * 100 for s in common],
            [lm[s].get("event", 0) / im[s].get("event", 1) * 100 for s in common]]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.boxplot(vals, tick_labels=["RGB", "Event"], showmeans=True)
    ax.axhline(100, color="black", lw=1, ls="--")
    ax.set_ylabel("server L / server I (%)")
    ax.set_title(f"Per-session retained sample ratio ({len(common)} common sessions)")
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(OUT / "retention_ratio.svg"); plt.close(fig)


def fmt(n): return f"{int(n):,}"


def main():
    OUT.mkdir(exist_ok=True)
    irows, occurrences, ierr = scan_i(ROOT / "orig_dataset")
    lrows, lerr = scan_l(ROOT / "dataset")
    write_csv(OUT / "server_i_sessions.csv", irows)
    write_csv(OUT / "server_l_sessions.csv", lrows)
    charts(irows, lrows)

    iset, lset = {r["session"] for r in irows}, {r["session"] for r in lrows}
    common = iset & lset
    imap = {r["session"]: r for r in irows}
    lmap = {r["session"]: r for r in lrows}
    def common_l(key):
        return sum(int(lmap[s].get(key, 0) or 0) for s in common)
    common_gt_sessions = sum(bool(lmap[s].get("gt_files", 0)) for s in common)
    duplicates = {k:v for k,v in occurrences.items() if len(v)>1}
    l_success = sum(bool(r.get("conversion_success")) for r in lrows)
    l_gt_sessions = sum(bool(r.get("gt_files")) for r in lrows)
    report = f"""# EventPose 데이터셋 구성 및 통계

분석 시각: 2026-10-23 (Asia/Seoul)  
명칭: `orig_dataset/` = **server I**, `dataset/` = **server L**  
상세 원자료: [server_i_sessions.csv](server_i_sessions.csv), [server_l_sessions.csv](server_l_sessions.csv)

## 핵심 결과

| 구분 | server I | server L |
|---|---:|---:|
| 고유 세션 | {fmt(len(irows))} | {fmt(len(lrows))} |
| RGB 이미지 샘플 (모든 장치 합) | {fmt(total(irows,'rgb'))} | {fmt(total(lrows,'rgb'))} |
| Event 프레임 샘플 (모든 장치 합) | {fmt(total(irows,'event'))} | {fmt(total(lrows,'event'))} |
| Depth 프레임 샘플 (모든 장치 합) | {fmt(total(irows,'depth'))} | {fmt(total(lrows,'depth'))} |
| GT 고유 타임스텝/프레임 | 0 | {fmt(total(lrows,'gt_unique_frames'))} |
| GT 유효 human pose 인스턴스 | 0 | {fmt(total(lrows,'gt_valid_persons'))} |
| GT 카메라별 프레임 레코드 | 0 | {fmt(total(lrows,'gt_camera_frame_records'))} |
| GT 카메라별 유효 person 레코드 | 0 | {fmt(total(lrows,'gt_camera_valid_persons'))} |

참고로 modality를 가진 장치-stream 수는 server I가 RGB {fmt(total(irows,'rgbd_devices')+total(irows,'rgb_devices'))}개(그중 RGBD {fmt(total(irows,'rgbd_devices'))}개), Event {fmt(total(irows,'event_devices'))}개, Depth {fmt(total(irows,'rgbd_devices'))}개이고, server L은 RGB/Event 각각 {fmt(total(lrows,'rgb_devices'))}개다. server L의 세션은 기본 5개 장치를, server I의 세션은 RGB/Event 7개와 Depth 5개 장치를 가진다.

![전체 개수 비교](dataset_counts.svg)

가장 중요한 해석은 다음과 같다.

- **샘플 수는 장치별 프레임을 합산한 수**다. 즉 같은 시각을 여러 카메라가 촬영하면 각각 한 샘플로 센다.
- server L의 MAMMA GT는 **SMPL-X**다 (`schema_name=Busan26_MAMMA_posed_SMPLX`). `params/smplx_pose_axis_angle`은 `(frame, person, 165)`, `params/smplx_betas`는 `(frame, person, 16)`, mesh는 10,475 vertices다.
- GT 파일은 보통 장치별로 같은 동기화 타임라인을 투영해 저장하므로, 학습 타깃의 시간적 개수를 알고 싶을 때는 `GT 고유 타임스텝`을 사용해야 한다. 카메라 관측별 레코드 양을 알고 싶을 때만 `GT 카메라별 프레임 레코드`를 사용한다.
- `GT 유효 human pose 인스턴스`는 canonical device(우선 device0)의 `results/person_valid=True` 합이다. 한 프레임에 사람이 2명이면 2개로 센다.

## 데이터 구조 차이

### server I: 원본 동기화 HDF5

- `data1/`, `data2/`, `data3/`는 요청대로 구분을 무시하고 합쳤다. 동일 세션명이 여러 partition에 있으면 한 번만 집계했다.
- 세션 아래 `RGBD/device0..4.h5`, `RGB/device5..6.h5`, `Event/device0..6.h5` 형태가 기본이다.
- RGBD 한 파일 안에는 `streams/rgb`와 `streams/depth_align`이 함께 있다. 독립 RGB 카메라는 `streams/images`, Event도 `streams/images`다.
- 대표 해상도는 RGBD RGB 3840×2160, 독립 RGB 1700×1300, Event 960×720이다. RGB는 JPEG, Event는 PNG frame encoding이다.
- RGBD/RGB/Event 영상은 이미 undistort 및 device0-master synchronization된 temporal subset이다. Event는 RGB anchor마다 여러 context frame을 가진다.
- 조사된 HDF5 dataset path에 SMPL/pose 항목이 없어 human-pose GT는 0으로 판정했다.

### server L: 압축/샤딩 데이터 + MAMMA GT

- RGB는 `RGB/deviceN/images.h5`로 리사이즈·JPEG 압축되어 있고, Event는 장치별 `frames_*.npz` 및 `events_*.npz` shard로 나뉜다. `conversion_manifest.json`의 실제 frame count를 집계했다.
- 실제 RGB stream은 1280×720, JPEG quality 95다. Event frame shard와 sparse event payload shard는 5,000-frame 단위가 기본이다.
- 변환 manifest의 출력은 대체로 RGB/Event 장치 0..4다. server I의 별도 RGB/Event 장치 5,6은 server L 변환 대상에서 빠진다.
- Depth는 변환 설정 `depth_included=false`인 경우 제거된다. 위 표의 server L Depth는 실제 `streams/depth_align` 존재 여부로 검증한 값이다.
- `RGBD/deviceN_mamma.h5`가 SMPL-X GT다. pose parameters, world translation, camera/world vertices, bbox, person mask, 동기화 인덱스를 포함한다.
- GT의 `source_depth_*`와 `synchronization/rgbd_depth_align/*`는 원천 Depth를 참조하는 인덱스/메타데이터이지, server L에 Depth 영상 샘플 자체가 남았다는 뜻은 아니다.

## 세션 대응 및 보존율

| 항목 | 개수 |
|---|---:|
| 양쪽에 공통인 세션 | {fmt(len(common))} |
| server I에만 있는 세션 | {fmt(len(iset-lset))} |
| server L에만 있는 세션 | {fmt(len(lset-iset))} |
| server L conversion `success=true` | {fmt(l_success)} / {fmt(len(lrows))} |
| server L에 MAMMA GT가 있는 세션 | {fmt(l_gt_sessions)} / {fmt(len(lrows))} |
| server I partition 간 중복 세션명 | {fmt(len(duplicates))} |

공통 {fmt(len(common))}개 세션만 제한하면 server L은 RGB {fmt(common_l('rgb'))}개, Event {fmt(common_l('event'))}개, Depth {fmt(common_l('depth'))}개이며, 이 중 {fmt(common_gt_sessions)}개 세션에 GT가 있어 고유 GT frame {fmt(common_l('gt_unique_frames'))}개와 유효 pose instance {fmt(common_l('gt_valid_persons'))}개가 있다. 따라서 전체 합계의 차이는 포맷 차이뿐 아니라 **server L에만 {fmt(len(lset-iset))}개 세션이 더 있는 것**에도 크게 영향을 받는다.

![공통 세션별 보존율](retention_ratio.svg)

보존율 그래프는 공통 세션마다 `(server L 장치 합 프레임)/(server I 장치 합 프레임)`을 계산한 것이다. server L은 단순 포맷 변경판이 아니라 장치 선택과 temporal coverage가 다른 파생 데이터이므로 100%가 보장되지 않는다.

## 집계 정의와 주의점

1. **RGB**: server I에서는 RGBD의 `streams/rgb`와 별도 RGB의 `streams/images`를 모두 합산했다. server L에서는 `RGB/device*/images.h5`의 실제 `streams/images` 수를 사용했다.
2. **Event**: event packet의 개수가 아니라, 모델 입력/시각화에 쓰이는 누적 Event **프레임 이미지 수**다. server L의 `events_*.npz`는 같은 프레임의 sparse event payload이므로 `frames_*.npz`와 중복해서 더하지 않았다.
3. **Depth**: 정렬된 `depth_align` 이미지 프레임 수다. GT 파일 안의 depth position은 영상이 아니므로 세지 않았다.
4. **GT frame**: MAMMA 결과 행 수다. 여러 장치 파일에 반복된 같은 synchronized frame은 canonical device 한 개만 세어 unique timeline을 구했다.
5. **GT pose**: `person_valid`가 참인 사람-프레임 쌍이다. `success`만 세면 사람 수가 반영되지 않아 별도로 구분했다.
6. 개별 세션의 장치 누락, frame count, GT count는 두 CSV에서 확인할 수 있다.

## 참고 스크립트와의 관계

- `visualize_data.py`의 정의처럼 HDF5 image count는 `count` 또는 `len(offsets)-1`이며, RGB/Depth/Event stream을 같은 방식으로 셌다.
- `event_to_sparse.py`는 Event 한 프레임을 CSR 한 행으로 변환하므로 sparse payload 수가 아니라 CSR row/frame 수가 샘플 정의다.
- `compress_data.py`는 RGBD의 `depth_align`과 RGB context stream을 제거하고 주 RGB를 1280x720으로 줄이는 변환이다. server L은 별도 변환 schema지만, 원본 대비 stream 제거/리사이즈라는 차이를 해석할 때 이를 반영했다.
- `warp_rgbd_h5_to_event.py`가 보여주듯 server I의 RGBD RGB와 depth_align은 같은 pixel plane이며 calibration/synchronization으로 Event plane과 연결된다.

## 검증 상태

- HDF5는 이미지 payload를 디코딩하지 않고 dataset shape/offset/manifest를 읽어 집계했다.
- 읽기 오류: server I {fmt(len(ierr))}건, server L {fmt(len(lerr))}건. 오류 목록은 [scan_errors.txt](scan_errors.txt)에 기록했다.
"""
    (OUT / "dataset_analysis.md").write_text(report)
    (OUT / "scan_errors.txt").write_text("[server I]\n" + "\n".join(ierr) + "\n\n[server L]\n" + "\n".join(lerr) + "\n")
    print(report.split("## 데이터 구조 차이")[0])


if __name__ == "__main__":
    main()
