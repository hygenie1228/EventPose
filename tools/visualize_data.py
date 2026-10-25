#!/usr/bin/env python3
"""
ETRI Multimodal Data Visualizer & Video Converter
=================================================
HDF5 데이터셋(RGB, Depth, Event) 구조 분석, 단일 프레임 이미지 추출,
그리고 VSCode 친화적 고속 MP4 비디오 변환을 단일 파일로 지원합니다.

사용 예시:
  1) MP4 비디오 변환 (기본값: 세션 내 Vis/ 폴더에 저장, VSCode 호환):
     python visualize_data.py --mode video

  2) 데이터 구조 및 HDF5 메타데이터 분석:
     python visualize_data.py --mode structure

  3) 특정 프레임 정지 이미지 시각화 (PNG 저장):
     python visualize_data.py --mode image --frame 100

  4) 부드러운 30fps 배속 재생 모드로 비디오 생성:
     python visualize_data.py --mode video --fps-mode smooth
"""

import argparse
import io
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
from PIL import Image

DATA_ROOT = Path(__file__).parent / "data"
DEFAULT_SESSION = "20260720_125338_040324_200_f"


# ==============================================================================
# 1. HDF5 데이터셋 구조 분석 (Structure Inspection)
# ==============================================================================
def print_data_structure(session_dir: Path):
    """세션 내 HDF5 파일들의 내부 데이터셋 구조 및 메타데이터 출력"""
    print("=" * 65)
    print(f"📊 DATA STRUCTURE ANALYSIS: {session_dir.name}")
    print("=" * 65)

    targets = [
        ("RGBD", "RGBD/device0.h5", ["streams/rgb", "streams/depth_align"]),
        ("Event", "Event/device0.h5", ["streams/images"]),
    ]

    for modality, rel_path, stream_names in targets:
        h5_path = session_dir / rel_path
        if not h5_path.exists():
            print(f"\n⚠️  {h5_path} 파일을 찾을 수 없습니다.")
            continue

        print(f"\n📁 [{modality}] — {h5_path.name}")
        with h5py.File(h5_path, "r") as f:
            print("  [Root Attributes]")
            for k in sorted(f.attrs.keys()):
                print(f"    - {k}: {f.attrs[k]}")

            for sname in stream_names:
                if sname in f:
                    grp = f[sname]
                    cnt = grp.attrs.get("count", len(grp["offsets"]) - 1 if "offsets" in grp else "unknown")
                    w = grp.attrs.get("width", "unknown")
                    h = grp.attrs.get("height", "unknown")
                    codec = grp.attrs.get("codec", "unknown")

                    print(f"\n  📊 Stream: {sname}")
                    print(f"    - 해상도: {w}x{h} | 프레임 수: {cnt} | 압축 코덱: {codec}")

                    if "timestamp_ns" in grp:
                        ts = grp["timestamp_ns"][:]
                        dur_sec = (ts[-1] - ts[0]) / 1e9 if len(ts) > 1 else 0
                        fps = len(ts) / dur_sec if dur_sec > 0 else 0
                        print(f"    - 총 촬영 시간: {dur_sec:.2f}초 | 실제 평균 FPS: {fps:.2f} fps")


# ==============================================================================
# 2. 고속 프레임 디코딩 (Fast Decoding Helper)
# ==============================================================================
def decode_h5_frame(stream: h5py.Group, idx: int, flag=cv2.IMREAD_COLOR) -> np.ndarray:
    """HDF5 바이트 스트림에서 단일 이미지를 C++ OpenCV로 초고속 디코딩"""
    offsets = stream["offsets"]
    s = int(offsets[idx])
    e = int(offsets[idx + 1])
    buf = stream["encoded_data"][s:e]
    return cv2.imdecode(np.frombuffer(buf, np.uint8), flag)


# ==============================================================================
# 3. 단일 프레임 정지 이미지 추출 및 시각화 (Image Mode)
# ==============================================================================
def extract_sample_images(session_dir: Path, frame_idx: int = 100, out_dir: Path = None):
    """RGB, Depth(컬러맵), Event 단일 프레임을 PNG 이미지로 저장"""
    if out_dir is None:
        out_dir = session_dir / "Vis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Image Mode] 프레임 {frame_idx} 정지 이미지 추출 중...")

    # 1. RGB
    rgb_h5 = session_dir / "RGBD" / "device0.h5"
    if rgb_h5.exists():
        with h5py.File(rgb_h5, "r") as f:
            rgb_bgr = decode_h5_frame(f["streams/rgb"], frame_idx, cv2.IMREAD_COLOR)
            out_p = out_dir / f"frame_{frame_idx:04d}_rgb.png"
            cv2.imwrite(str(out_p), rgb_bgr)
            print(f"  ✅ RGB 저장: {out_p.name} ({rgb_bgr.shape[1]}x{rgb_bgr.shape[0]})")

    # 2. Depth (Turbo Colormap)
    if rgb_h5.exists():
        with h5py.File(rgb_h5, "r") as f:
            depth_16u = decode_h5_frame(f["streams/depth_align"], frame_idx, cv2.IMREAD_UNCHANGED)
            valid = depth_16u > 0
            scaled = np.clip((depth_16u.astype(np.float32) / 5000.0) * 255.0, 0, 255).astype(np.uint8)
            color_depth = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
            color_depth[~valid] = 0
            out_p = out_dir / f"frame_{frame_idx:04d}_depth.png"
            cv2.imwrite(str(out_p), color_depth)
            print(f"  ✅ Depth 저장: {out_p.name} ({depth_16u.shape[1]}x{depth_16u.shape[0]})")

    # 3. Event
    event_h5 = session_dir / "Event" / "device0.h5"
    if event_h5.exists():
        with h5py.File(event_h5, "r") as f:
            # Event는 50배 더 많으므로 인덱스 스케일링
            ev_idx = min(frame_idx * 50, len(f["streams/images/offsets"]) - 2)
            ev_img = decode_h5_frame(f["streams/images"], ev_idx, cv2.IMREAD_GRAYSCALE)
            out_p = out_dir / f"frame_{ev_idx:06d}_event.png"
            cv2.imwrite(str(out_p), ev_img)
            print(f"  ✅ Event 저장: {out_p.name} ({ev_img.shape[1]}x{ev_img.shape[0]})")


# ==============================================================================
# 4. 고속 MP4 비디오 변환 (Video / MP4 Mode)
# ==============================================================================
def convert_stream_to_mp4(
    h5_path: str,
    stream_name: str,
    out_path: str,
    fps: int,
    scale_w: int = None,
    scale_h: int = None,
    crf: int = 22,
    is_depth: bool = False,
    is_event: bool = False,
):
    """HDF5 스트림을 OpenCV로 실시간 디코딩 후 FFmpeg 파이프로 초고속 MP4 인코딩"""
    with h5py.File(h5_path, "r") as f:
        stream = f[stream_name]
        count = int(stream.attrs["count"])
        offsets = stream["offsets"]
        enc_data = stream["encoded_data"]

        def frames():
            for i in range(count):
                buf = enc_data[int(offsets[i]) : int(offsets[i + 1])]
                if is_depth:
                    d16 = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_UNCHANGED)
                    valid = d16 > 0
                    scaled = np.clip((d16.astype(np.float32) / 5000.0) * 255.0, 0, 255).astype(np.uint8)
                    frame = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
                    frame[~valid] = 0
                    yield frame
                elif is_event:
                    yield cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_GRAYSCALE)
                else:
                    yield cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)

        encode_frames_to_mp4(
            frames(), count, int(stream.attrs["width"]), int(stream.attrs["height"]),
            out_path, fps, scale_w, scale_h, crf, is_gray=is_event,
        )


def convert_event_npz_to_mp4(npz_path: str, out_path: str, fps: int, scale_w=None, scale_h=None, crf: int = 28):
    """sparse CSR npz (event_to_sparse.py 출력) 를 MP4 로 변환"""
    ev = np.load(npz_path)
    h, w = int(ev["frame_height"]), int(ev["frame_width"])
    indptr, indices, data = ev["indptr"], ev["indices"], ev["data"]
    count = len(indptr) - 1
    vals = np.where(data > 0, 255, 0).astype(np.uint8)

    def frames():
        for i in range(count):
            img = np.full(h * w, 128, np.uint8)
            img[indices[indptr[i] : indptr[i + 1]]] = vals[indptr[i] : indptr[i + 1]]
            yield img.reshape(h, w)

    encode_frames_to_mp4(frames(), count, w, h, out_path, fps, scale_w, scale_h, crf, is_gray=True)


def encode_frames_to_mp4(frames, count, orig_w, orig_h, out_path, fps, scale_w=None, scale_h=None, crf=22, is_gray=False):
    """프레임 이터레이터를 FFmpeg 파이프로 MP4 인코딩"""
    name_label = Path(out_path).stem.upper()
    print(f"\n[{name_label}] 변환 시작 -> {out_path} ({fps} FPS)")

    out_w = scale_w if scale_w else orig_w
    out_h = scale_h if scale_h else orig_h
    need_resize = (out_w != orig_w) or (out_h != orig_h)

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "gray" if is_gray else "bgr24",
        "-s", f"{out_w}x{out_h}",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        out_path,
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    t_start = time.time()
    log_step = 500 if is_gray else 50

    for i, frame in enumerate(frames):
        if need_resize:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

        proc.stdin.write(frame.tobytes())

        if (i + 1) % log_step == 0 or i == count - 1:
            elapsed = time.time() - t_start
            cur_fps = (i + 1) / elapsed
            remain = (count - (i + 1)) / cur_fps if cur_fps > 0 else 0
            print(
                f"\r  {name_label}: [{i+1}/{count}] ({((i+1)/count)*100:.1f}%) "
                f"| {cur_fps:.1f} fps | 남은 시간: {remain:.0f}s",
                end="",
                flush=True,
            )

    proc.stdin.close()
    proc.wait()
    print()

    size_mb = Path(out_path).stat().st_size / (1024 * 1024)
    print(f"  ✅ {name_label} 완료: {out_path} ({size_mb:.1f} MB, 소요시간: {time.time()-t_start:.1f}s)")


def convert_all_videos(
    session_dir: Path,
    out_dir: Path,
    fps_mode: str = "real",
    resolution: str = "small",
    modalities: list = None,
):
    """RGB, Depth, Event 비디오 일괄 변환 (해상도 및 용량 최적화)"""
    if modalities is None:
        modalities = ["rgb", "depth", "event"]

    out_dir.mkdir(parents=True, exist_ok=True)

    # FPS 모드 설정
    if fps_mode == "real":
        rgb_fps, depth_fps, event_fps = 5, 5, 250
    else:  # smooth (자연스러운 30fps 배속)
        rgb_fps, depth_fps, event_fps = 30, 30, 250

    # 해상도 프리셋 설정 (용량 및 VSCode 재생 속도 최적화)
    if resolution == "tiny":      # 극초경량 (1~10MB)
        rgb_w, rgb_h, rgb_crf = 480, 270, 28
        depth_w, depth_h, depth_crf = 480, 270, 28
        ev_w, ev_h, ev_crf = 320, 240, 30
    elif resolution == "medium":  # 720p 중간
        rgb_w, rgb_h, rgb_crf = 1280, 720, 22
        depth_w, depth_h, depth_crf = 1280, 720, 22
        ev_w, ev_h, ev_crf = 640, 480, 26
    elif resolution == "orig":    # 원본 해상도 (4K)
        rgb_w, rgb_h, rgb_crf = None, None, 20
        depth_w, depth_h, depth_crf = None, None, 20
        ev_w, ev_h, ev_crf = None, None, 24
    else:  # small (기본값: 360p, 초경량 3~15MB)
        rgb_w, rgb_h, rgb_crf = 640, 360, 26
        depth_w, depth_h, depth_crf = 640, 360, 26
        ev_w, ev_h, ev_crf = 480, 360, 28

    print("=" * 65)
    print(f"🎬 HDF5 -> MP4 변환 시작")
    print(f"세션: {session_dir.name}")
    print(f"출력 폴더: {out_dir.resolve()}")
    print(f"해상도 프리셋: [{resolution}] (RGB/Depth: {rgb_w}x{rgb_h}, Event: {ev_w}x{ev_h})")
    print(f"FPS 모드: {fps_mode} (RGB/Depth: {rgb_fps}fps, Event: {event_fps}fps)")
    print("=" * 65)

    # 1. RGB
    if "rgb" in modalities:
        convert_stream_to_mp4(
            str(session_dir / "RGBD" / "device0.h5"),
            "streams/rgb",
            str(out_dir / "rgb.mp4"),
            fps=rgb_fps,
            scale_w=rgb_w,
            scale_h=rgb_h,
            crf=rgb_crf,
        )

    # 2. Depth (Turbo Colormap)
    if "depth" in modalities:
        convert_stream_to_mp4(
            str(session_dir / "RGBD" / "device0.h5"),
            "streams/depth_align",
            str(out_dir / "depth.mp4"),
            fps=depth_fps,
            scale_w=depth_w,
            scale_h=depth_h,
            crf=depth_crf,
            is_depth=True,
        )

    # 3. Event
    if "event" in modalities and (session_dir / "Event" / "device0.npz").exists():
        convert_event_npz_to_mp4(
            str(session_dir / "Event" / "device0.npz"),
            str(out_dir / "event.mp4"),
            fps=event_fps,
            scale_w=ev_w,
            scale_h=ev_h,
            crf=ev_crf,
        )
    elif "event" in modalities:
        convert_stream_to_mp4(
            str(session_dir / "Event" / "device0.h5"),
            "streams/images",
            str(out_dir / "event.mp4"),
            fps=event_fps,
            scale_w=ev_w,
            scale_h=ev_h,
            crf=ev_crf,
            is_event=True,
        )

    print("\n" + "=" * 65)
    print(f"🎉 모든 변환 완료! 최종 비디오 파일 목록 ({out_dir}):")
    for f in sorted(out_dir.glob("*.mp4")):
        mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name} ({mb:.1f} MB)")
    print("=" * 65)


# ==============================================================================
# 메인 CLI 인터페이스
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="ETRI Multimodal Data Visualizer & Fast MP4 Converter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["video", "image", "structure"],
        default="video",
        help="실행 모드: video (MP4 변환, 기본값) | image (정지 이미지 추출) | structure (데이터 구조 분석)",
    )
    parser.add_argument("--session", default=DEFAULT_SESSION, help=f"세션 디렉토리 이름 (기본값: {DEFAULT_SESSION})")
    parser.add_argument("--data-root", default=None, help="데이터 루트 경로 (기본값: ./data)")
    parser.add_argument("--out-dir", default=None, help="결과물 저장 경로 (기본값: 세션 내부 Vis/)")
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=["rgb", "depth", "event"],
        choices=["rgb", "depth", "event"],
        help="처리할 모달리티 선택 (기본값: rgb depth event)",
    )
    parser.add_argument(
        "--fps-mode",
        choices=["real", "smooth"],
        default="real",
        help="비디오 FPS 모드: real (실제 5fps/250fps 동기화) | smooth (30fps 배속 재생)",
    )
    parser.add_argument(
        "--res",
        "--resolution",
        dest="resolution",
        choices=["small", "tiny", "medium", "orig"],
        default="small",
        help="해상도/용량 프리셋: small (기본값: 360p, 초경량 3~15MB) | tiny (270p, 극초경량 1~10MB) | medium (720p) | orig (원본 4K)",
    )
    parser.add_argument("--frame", type=int, default=100, help="image 모드 시 추출할 프레임 번호 (기본값: 100)")

    args = parser.parse_args()

    data_root = Path(args.data_root) if args.data_root else DATA_ROOT
    session_dir = data_root / args.session

    if not session_dir.exists():
        print(f"❌ 세션 디렉토리를 찾을 수 없습니다: {session_dir}")
        sys.exit(1)

    out_dir = Path(args.out_dir) if args.out_dir else session_dir / "Vis"

    if args.mode == "structure":
        print_data_structure(session_dir)
    elif args.mode == "image":
        extract_sample_images(session_dir, frame_idx=args.frame, out_dir=out_dir)
    elif args.mode == "video":
        convert_all_videos(
            session_dir,
            out_dir,
            fps_mode=args.fps_mode,
            resolution=args.resolution,
            modalities=args.modalities,
        )


if __name__ == "__main__":
    main()
