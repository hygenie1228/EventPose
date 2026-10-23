#!/usr/bin/env python3
"""
Event/deviceN.h5 (PNG 프레임, 값 0/128/255) 를 sparse CSR npz 로 변환합니다.

npz 는 scipy.sparse.save_npz 와 동일한 키를 가지므로 두 방식 모두로 읽을 수 있습니다:
    m = scipy.sparse.load_npz("device0.npz")          # (N, H*W) int8 CSR, 값 -1/+1
    frame = m[i].toarray().reshape(H, W)
또는 scipy 없이:
    from event_to_sparse import load_event_npz, event_frame
    ev = load_event_npz("device0.npz"); frame = event_frame(ev, i)   # 0/128/255 uint8

CSR 외의 키 (h5 의 경로 '/' 를 '__' 로 바꾼 이름):
    streams__images__timestamp_ns 등  : 프레임별 배열 그대로
    synchronization__*, calibration__*: 그대로
    *_json                           : JSON 문자열 (numpy unicode 스칼라)
    attrs_json                       : h5 의 모든 attrs {경로: {키: 값}}
    frame_height, frame_width

사용 예시:
  python event_to_sparse.py --root exp/compressed            # 모든 세션, 변환 후 h5 삭제
  python event_to_sparse.py --root exp/compressed --keep-h5
"""

import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import h5py
import numpy as np

NO_EVENT = 128


def _jsonable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, bytes):
        return v.decode()
    return v


def _key(path: str) -> str:
    return path.strip("/").replace("/", "__")


def load_event_npz(path):
    return dict(np.load(path))


def event_frame(ev: dict, i: int) -> np.ndarray:
    """i 번째 프레임을 원본과 같은 0/128/255 uint8 이미지로 복원"""
    h, w = int(ev["frame_height"]), int(ev["frame_width"])
    s, e = ev["indptr"][i], ev["indptr"][i + 1]
    img = np.full(h * w, NO_EVENT, np.uint8)
    img[ev["indices"][s:e]] = np.where(ev["data"][s:e] > 0, 255, 0).astype(np.uint8)
    return img.reshape(h, w)


def convert(h5_path: Path, out_path: Path, keep_h5: bool):
    t0 = time.time()
    arrays, attrs = {}, {}
    with h5py.File(h5_path, "r") as f:
        g = f["streams/images"]
        offsets = g["offsets"][:]
        n = len(offsets) - 1
        h, w = int(g.attrs["height"]), int(g.attrs["width"])

        indptr = np.zeros(n + 1, np.int64)
        idx_list, val_list = [], []
        enc = g["encoded_data"]
        step = 4096
        for b in range(0, n, step):
            e_ = min(b + step, n)
            buf = enc[int(offsets[b]) : int(offsets[e_])]
            for i in range(b, e_):
                s = int(offsets[i] - offsets[b])
                img = cv2.imdecode(buf[s : s + int(offsets[i + 1] - offsets[i])], cv2.IMREAD_GRAYSCALE).ravel()
                nz = np.flatnonzero(img != NO_EVENT).astype(np.int32)
                vals = img[nz]
                assert np.isin(vals, (0, 255)).all(), f"{h5_path} frame {i}: unexpected values {np.unique(vals)}"
                idx_list.append(nz)
                val_list.append(np.where(vals == 255, 1, -1).astype(np.int8))
                indptr[i + 1] = indptr[i] + nz.size

        def visit(name, obj):
            attrs[name] = {k: _jsonable(v) for k, v in obj.attrs.items()}
            if not isinstance(obj, h5py.Dataset) or name in ("streams/images/encoded_data", "streams/images/offsets"):
                return
            v = obj[()]
            if isinstance(v, bytes):
                arrays[_key(name)] = np.str_(v.decode())
            elif obj.dtype.kind == "O":
                arrays[_key(name)] = np.array([x.decode() if isinstance(x, bytes) else x for x in v])
            else:
                arrays[_key(name)] = v

        attrs["/"] = {k: _jsonable(v) for k, v in f.attrs.items()}
        f.visititems(visit)

    indices = np.concatenate(idx_list) if idx_list else np.zeros(0, np.int32)
    data = np.concatenate(val_list) if val_list else np.zeros(0, np.int8)
    tmp = out_path.with_name(out_path.stem + ".tmp.npz")
    np.savez_compressed(
        tmp,
        format=np.array(b"csr"),
        shape=np.array([n, h * w]),
        indptr=indptr,
        indices=indices,
        data=data,
        frame_height=np.array(h),
        frame_width=np.array(w),
        attrs_json=np.str_(json.dumps(attrs)),
        **arrays,
    )

    # 검증: 저장한 npz 를 다시 읽어 CSR 이 같고, 임의 프레임이 원본 PNG 와 같은지 확인
    ev = load_event_npz(tmp)
    assert np.array_equal(ev["indptr"], indptr) and np.array_equal(ev["indices"], indices) and np.array_equal(ev["data"], data)
    with h5py.File(h5_path, "r") as f:
        g = f["streams/images"]
        for i in np.random.default_rng(0).choice(n, size=min(20, n), replace=False):
            ref = cv2.imdecode(g["encoded_data"][int(offsets[i]) : int(offsets[i + 1])], cv2.IMREAD_GRAYSCALE)
            assert np.array_equal(event_frame(ev, i), ref), f"{h5_path} frame {i} mismatch"
    tmp.rename(out_path)

    sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    out_path.with_name(out_path.name + ".sha256").write_text(f"{sha}  {out_path.name}\n")
    old = h5_path.stat().st_size
    if not keep_h5:
        h5_path.unlink()
        h5_path.with_name(h5_path.name + ".sha256").unlink(missing_ok=True)
    return f"{out_path}  ({old/1e9:.2f}GB -> {out_path.stat().st_size/1e9:.3f}GB, {n} frames, {time.time()-t0:.0f}s)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).parent / "exp" / "compressed"))
    ap.add_argument("--sessions", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--keep-h5", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    sessions = args.sessions or sorted(p.name for p in root.iterdir() if (p / "Event").is_dir())
    jobs = [(h, h.with_suffix(".npz")) for s in sessions for h in sorted((root / s / "Event").glob("*.h5"))]
    print(f"{len(jobs)} event h5 files", flush=True)

    failed = 0
    with ProcessPoolExecutor(args.workers) as ex:
        futs = {ex.submit(convert, h, o, args.keep_h5): h for h, o in jobs}
        for f in as_completed(futs):
            try:
                print("[done]", f.result(), flush=True)
            except Exception as e:
                failed += 1
                print("[FAILED]", futs[f], repr(e), flush=True)
    print(f"ALL DONE, failed={failed}", flush=True)


if __name__ == "__main__":
    main()
