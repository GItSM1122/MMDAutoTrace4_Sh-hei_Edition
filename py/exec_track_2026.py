"""MediaPipe-based 2026 tracking path for MMDAutoTrace4.

This avoids legacy PHALP/HMR2/Detectron2 model downloads. It reads a video,
runs MediaPipe Pose Landmarker directly, maps its landmarks into the 49-joint
layout expected by exec_pkl2json.py, and writes a compatible joblib PKL.
"""

import os
import cv2
import joblib
import numpy as np
import mediapipe as mp


# Indices in the 49-joint layout used by exec_pkl2json.py.
# Values are MediaPipe Pose landmark indices.
DIRECT = {
    0: 0, 2: 12, 3: 14, 4: 16, 5: 11, 6: 13, 7: 15,
    9: 24, 10: 26, 11: 28, 12: 23, 13: 25, 14: 27,
    15: 5, 16: 2, 17: 8, 18: 7,
    19: 31, 21: 29, 22: 32, 24: 30,
    25: 28, 26: 26, 27: 24, 28: 23, 29: 25, 30: 27,
    31: 16, 32: 14, 33: 12, 34: 11, 35: 13, 36: 15,
    44: 0, 45: 2, 46: 5, 47: 7, 48: 8,
}


def _avg(a, b):
    return (a + b) * 0.5


def _map_world(world):
    src = np.array([[p.x, p.y, p.z] for p in world], dtype=np.float32)
    dst = np.zeros((49, 3), dtype=np.float32)
    for di, si in DIRECT.items():
        dst[di] = src[si]

    # OpenPose-derived/synthetic joints.
    dst[1] = _avg(src[11], src[12])                 # neck
    dst[8] = _avg(src[23], src[24])                 # mid hip
    dst[20] = src[31]                               # left small toe approximation
    dst[23] = src[32]                               # right small toe approximation
    dst[37] = dst[1]                                # neck LSP
    dst[38] = src[0]                                # top head approximation
    dst[39] = dst[8]                                # pelvis
    dst[40] = dst[1]                                # thorax
    dst[41] = _avg(dst[8], dst[1])                  # spine
    dst[42] = _avg(src[9], src[10])                 # jaw approximation
    dst[43] = src[0]                                # head
    return dst


def _map_image(image_landmarks, width, height):
    src = np.array([[p.x * width, p.y * height] for p in image_landmarks], dtype=np.float32)
    dst = np.zeros((49, 2), dtype=np.float32)
    for di, si in DIRECT.items():
        dst[di] = src[si]
    dst[1] = _avg(src[11], src[12])
    dst[8] = _avg(src[23], src[24])
    dst[20] = src[31]
    dst[23] = src[32]
    dst[37] = dst[1]
    dst[38] = src[0]
    dst[39] = dst[8]
    dst[40] = dst[1]
    dst[41] = _avg(dst[8], dst[1])
    dst[42] = _avg(src[9], src[10])
    dst[43] = src[0]
    return dst


def run(video_path: str, output_dir: str) -> None:
    print("Start: MediaPipe 2026 tracker =============================")
    if not os.path.isfile(video_path):
        raise FileNotFoundError(video_path)
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join("data", "pose_landmarker_full.task")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"MediaPipe model not found: {model_path}. "
            "Expected data/pose_landmarker_full.task"
        )

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = {}
    out_index = 0
    with PoseLandmarker.create_from_options(options) as detector:
        frame_id = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # Sample to approximately 30 fps.
            target_index = int(round(frame_id * 30.0 / fps))
            if target_index < out_index:
                frame_id += 1
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(round(frame_id * 1000.0 / fps))
            result = detector.detect_for_video(image, timestamp_ms)

            if result.pose_world_landmarks and result.pose_landmarks:
                joints3d = _map_world(result.pose_world_landmarks[0])
                joints2d = _map_image(result.pose_landmarks[0], width, height)
                xs, ys = joints2d[:, 0], joints2d[:, 1]
                x1, y1 = float(np.min(xs)), float(np.min(ys))
                x2, y2 = float(np.max(xs)), float(np.max(ys))
                bbox = np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)
                vis = np.array([p.visibility for p in result.pose_landmarks[0]], dtype=np.float32)
                conf = np.array(float(np.mean(vis)), dtype=np.float32)

                frames[out_index] = {
                    "time": out_index,
                    "tracked_ids": [0],
                    "tracked_bbox": [bbox],
                    "conf": [conf],
                    "camera": [np.zeros(3, dtype=np.float32)],
                    "3d_joints": [joints3d],
                    "2d_joints": [joints2d],
                }
            out_index += 1
            frame_id += 1

    cap.release()
    if not frames:
        raise RuntimeError("MediaPipe detected no pose frames.")

    pkl_path = os.path.join(output_dir, "00000_mediapipe.pkl")
    joblib.dump(frames, pkl_path)
    print("Pose frames:", len(frames))
    print("PKL:", pkl_path)

    # Convert immediately using the existing MMDAutoTrace4 converter.
    import exec_pkl2json
    exec_pkl2json.main(output_dir)
    print("PKL -> original JSON conversion complete.")
    print("End: MediaPipe 2026 tracker ===============================")
