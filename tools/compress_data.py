#!/usr/bin/env python3
"""
data/ 의 세션들을 압축 버전으로 exp/compressed/ 에 생성합니다.

- RGBD/deviceN.h5 : rgb 스트림을 1280x720 으로 리사이즈, depth_align 스트림 삭제
- RGB/deviceN.h5  : images 스트림을 1280x720 으로 리사이즈, context_t_minus_1/2 스트림 삭제
- Event/*.h5, metadata 등 : 그대로 복사 (Vis/, 세션 루트의 depth.mp4 는 제외)
  리사이즈한 h5 의 .sha256 은 새로 계산해서 씁니다.

리사이즈 시 calibration/rgb/{intrinsic_matrix, source_intrinsic_matrix} 를
x/y 스케일에 맞게 갱신합니다 (RGB 1700x1300 -> 1280x720 은 가로세로 비율이 달라 fx/fy 가 다르게 스케일됨).

사용 예시:
  python compress_data.py                    # data/ 의 모든 세션
  python compress_data.py --sessions 20260720_125338_040324_200_f
"""

import argparse
import hashlib
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import h5py
import numpy as np

ROOT = Path(__file__).parent
TARGET_WH = (1280, 720)
JPEG_QUALITY = 95
DROP_STREAMS = {"RGBD": {"depth_align"}, "RGB": {"context_t_minus_1", "context_t_minus_2"}}
SKIP_TOP = {"Vis", "depth.mp4"}


def scale_K(K: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """픽셀 중심 규약(cv2.resize)에 맞춘 intrinsic 스케일"""
    K = K.astype(np.float64).copy()
    K[0, 0] *= sx
    K[0, 1] *= sx
    K[0, 2] = (K[0, 2] + 0.5) * sx - 0.5
    K[1, 1] *= sy
    K[1, 2] = (K[1, 2] + 0.5) * sy - 0.5
    return K


def copy_attrs(src, dst):
    for k, v in src.attrs.items():
        dst.attrs[k] = v


def copy_tree(src: h5py.Group, dst: h5py.Group, skip):
    """src 그룹을 dst 로 재귀 복사 (skip(path) 가 True 인 항목 제외)"""
    copy_attrs(src, dst)
    for name, obj in src.items():
        if skip(obj.name):
            continue
        if isinstance(obj, h5py.Group):
            copy_tree(obj, dst.create_group(name), skip)
        else:
            src.copy(obj, dst, name=name)


def resize_stream(src: h5py.Group, dst: h5py.Group):
    offsets = src["offsets"][:]
    enc = src["encoded_data"]
    n = len(offsets) - 1
    orig_w, orig_h = int(src.attrs["width"]), int(src.attrs["height"])

    new_offsets = np.zeros(n + 1, dtype=np.uint64)
    chunks = []
    for i in range(n):
        buf = enc[int(offsets[i]) : int(offsets[i + 1])]
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        img = cv2.resize(img, TARGET_WH, interpolation=cv2.INTER_AREA)
        ok, out = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        assert ok
        chunks.append(out.ravel())
        new_offsets[i + 1] = new_offsets[i] + out.size

    data = np.concatenate(chunks) if chunks else np.zeros(0, np.uint8)
    dst.create_dataset("encoded_data", data=data, chunks=(min(len(data), 1 << 26),) if len(data) else None)
    dst.create_dataset("offsets", data=new_offsets)
    dst.attrs["width"] = TARGET_WH[0]
    dst.attrs["height"] = TARGET_WH[1]
    dst.attrs["encoded_byte_count"] = np.int64(len(data))
    dst.attrs["resized_from_wh"] = np.array([orig_w, orig_h], dtype=np.int64)

    meta = json.loads(src["processing_metadata_json"][()])
    sx, sy = TARGET_WH[0] / orig_w, TARGET_WH[1] / orig_h
    meta["resize"] = {
        "from_wh": [orig_w, orig_h],
        "to_wh": list(TARGET_WH),
        "interpolation": "INTER_AREA",
        "jpeg_quality": JPEG_QUALITY,
        "K_undistorted_resized": scale_K(np.array(meta["K_undistorted"]), sx, sy).tolist(),
    }
    meta["size_wh"] = list(TARGET_WH)
    del dst["processing_metadata_json"]
    dst.create_dataset("processing_metadata_json", data=json.dumps(meta), dtype=h5py.string_dtype())
    return orig_w, orig_h


def write_sha256(path: Path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    path.with_name(path.name + ".sha256").write_text(f"{h.hexdigest()}  {path.name}\n")


def process_h5(src_path: Path, dst_path: Path, modality: str):
    t0 = time.time()
    tmp = dst_path.with_suffix(".h5.tmp")
    drop = DROP_STREAMS[modality]
    with h5py.File(src_path, "r") as fs, h5py.File(tmp, "w") as fd:
        stream_names = [s for s in fs["streams"] if s not in drop]

        def skip(path):
            parts = path.strip("/").split("/")
            if parts[0] == "streams" and len(parts) >= 2:
                if parts[1] in drop:
                    return True
                if len(parts) == 3 and parts[2] in ("encoded_data", "offsets"):
                    return True
            return False

        copy_tree(fs, fd, skip)

        sizes = {resize_stream(fs[f"streams/{s}"], fd[f"streams/{s}"]) for s in stream_names}
        assert len(sizes) == 1, f"streams have different sizes: {sizes}"
        (orig_w, orig_h), = sizes
        sx, sy = TARGET_WH[0] / orig_w, TARGET_WH[1] / orig_h
        for k in ("intrinsic_matrix", "source_intrinsic_matrix"):
            ds = fd[f"calibration/rgb/{k}"]
            ds[...] = scale_K(ds[:], sx, sy)
            ds.attrs["resized_from_wh"] = np.array([orig_w, orig_h], dtype=np.int64)
            ds.attrs["resized_to_wh"] = np.array(TARGET_WH, dtype=np.int64)

        fd.attrs["resized_rgb_wh"] = np.array(TARGET_WH, dtype=np.int64)
        if drop:
            fd.attrs["dropped_streams"] = ",".join(sorted(drop))
    tmp.rename(dst_path)
    write_sha256(dst_path)
    return f"{dst_path}  ({src_path.stat().st_size/1e9:.2f}GB -> {dst_path.stat().st_size/1e9:.2f}GB, {time.time()-t0:.0f}s)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(ROOT / "data"))
    ap.add_argument("--out-root", default=str(ROOT / "exp" / "compressed"))
    ap.add_argument("--sessions", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()

    data_root, out_root = Path(args.data_root), Path(args.out_root)
    sessions = args.sessions or sorted(p.name for p in data_root.iterdir() if (p / "DEVICE0_MASTER_SYNC_COMPLETE.json").exists())

    jobs = []
    for s in sessions:
        src, dst = data_root / s, out_root / s
        dst.mkdir(parents=True, exist_ok=True)
        for item in src.iterdir():
            if item.name in SKIP_TOP:
                continue
            if item.name in DROP_STREAMS:  # RGBD / RGB
                (dst / item.name).mkdir(exist_ok=True)
                for h in sorted(item.glob("*.h5")):
                    if not (dst / item.name / h.name).exists():
                        jobs.append((h, dst / item.name / h.name, item.name))
                for other in item.iterdir():
                    if other.suffix not in (".h5", ".sha256") and not (dst / item.name / other.name).exists():
                        shutil.copy2(other, dst / item.name / other.name)
            elif item.is_dir():
                shutil.copytree(item, dst / item.name, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dst / item.name)
        print(f"[copied] {s}", flush=True)

    print(f"{len(jobs)} h5 files to resize", flush=True)
    failed = 0
    with ProcessPoolExecutor(args.workers) as ex:
        futs = {ex.submit(process_h5, *j): j for j in jobs}
        for f in as_completed(futs):
            try:
                print("[done]", f.result(), flush=True)
            except Exception as e:
                failed += 1
                print("[FAILED]", futs[f][0], repr(e), flush=True)
    print(f"ALL DONE, failed={failed}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
