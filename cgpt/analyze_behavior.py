#!/usr/bin/env python3

import os
import sys
import argparse
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
import eyed3
from moviepy.editor import VideoFileClip
from pyAudioAnalysis import audioBasicIO, ShortTermFeatures
import librosa


# ========== AUDIO EXTRACTION ==========
def extract_audio(video_path, audio_path="temp_audio.wav"):
    """Extract audio track from video using MoviePy"""
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path, verbose=False, logger=None)
    return audio_path


# ========== AUDIO ANALYSIS ==========
def analyze_audio(audio_path):
    """Compute basic prosodic and tonal features"""

    [Fs, x] = audioBasicIO.read_audio_file(audio_path)

    # Ensure mono
    if x.ndim > 1:
        x = np.mean(x, axis=1)

    win = int(0.05 * Fs)
    step = int(0.025 * Fs)

    if len(x) < win:
        x = np.pad(x, (0, win - len(x)), mode="constant")

    try:
        F, f_names = ShortTermFeatures.feature_extraction(x, Fs, win, step)
    except ValueError:
        win = int(0.04 * Fs)
        step = int(0.02 * Fs)
        F, f_names = ShortTermFeatures.feature_extraction(x, Fs, win, step)

    features = dict(zip(f_names, F.mean(axis=1)))
    # Silence ratio from ZCR
    features["silence_ratio"] = np.mean(F[f_names.index("zcr"), :] < 0.01)

    return features


# ========== VIDEO ANALYSIS ==========
def analyze_video(video_path):
    """Extract posture and facial movement metrics using MediaPipe"""
    mp_pose = mp.solutions.pose.Pose()
    mp_face = mp.solutions.face_mesh.FaceMesh(max_num_faces=1)
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # fallback

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = int(fps * 2)  # sample every 2 seconds

    rows = []
    frame_idx = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        if frame_idx % interval == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose = mp_pose.process(rgb)
            face = mp_face.process(rgb)

            row = {
                "frame": frame_idx,
                "time_sec": frame_idx / fps
            }

            # ---- posture features ----
            if pose.pose_landmarks:
                lm = pose.pose_landmarks.landmark
                ls, rs, lh, rh = lm[11], lm[12], lm[23], lm[24]

                shoulder_slope = abs(ls.y - rs.y)
                spine_angle = np.arctan2(abs(lh.y - ls.y), abs(lh.x - ls.x))

                row.update({
                    "shoulder_slope": shoulder_slope,
                    "spine_angle": spine_angle,
                })

            # ---- facial features ----
            if face.multi_face_landmarks:
                lm_face = face.multi_face_landmarks[0].landmark

                left_eye = np.mean([lm_face[i].y for i in [159, 145]])
                mouth_gap = abs(lm_face[13].y - lm_face[14].y)

                row.update({
                    "eye_openness": left_eye,
                    "mouth_gap": mouth_gap,
                })

            rows.append(row)

        frame_idx += 1

    cap.release()
    return pd.DataFrame(rows)


# ========== SUMMARY ==========
def summarize_results(video_path, window_seconds=10):
    audio_path = extract_audio(video_path)
    audio_features = analyze_audio(audio_path)
    video_features = analyze_video(video_path)

    if video_features.empty:
        print("⚠️ No video landmarks detected")
        return pd.DataFrame()

    # Convert seconds → window index (e.g., 0–9s → window 0, 10–19s → window 1)
    video_features["window"] = (video_features["time_sec"] // window_seconds).astype(int)

    summary = video_features.groupby("window").agg({
        "shoulder_slope": "mean",
        "spine_angle": "mean",
        "eye_openness": "mean",
        "mouth_gap": ["mean", "std"],
    })
    summary.columns = ["_".join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()
    summary.rename(columns={"window": "time_window"}, inplace=True)


    for k, v in audio_features.items():
        summary[k] = v

    os.remove(audio_path)
    return summary.reset_index()


def detect_behavior_violations(df, max_fraction=0.1):
    """
    Flag windows where metrics cross thresholds,
    but only keep the worst `max_fraction` of those windows per metric.

    max_fraction=0.3 → at most ~30% of windows per metric are flagged.
    """

    # Direction: 'above' = bad if higher than threshold
    #            'below' = bad if lower than threshold
    thresholds = {
        "shoulder_slope_mean": ("above", 0.03),
        "spine_angle_mean":    ("above", 1.40),
        "eye_openness_mean":   ("below", 0.26),
        "mouth_gap_mean":      ("above", 0.003),
    }

    results = {}

    for metric, (direction, base_thresh) in thresholds.items():
        series = df[metric].astype(float)

        if direction == "above":
            # First, keep only windows that cross the base threshold
            candidates = series[series > base_thresh]
            if candidates.empty:
                results[metric] = {"windows": [], "values": [], "threshold": base_thresh, "direction": direction}
                continue

            # Among those, keep only the top `max_fraction` worst values
            dyn_thresh = candidates.quantile(1 - max_fraction)
            final_thresh = max(base_thresh, dyn_thresh)
            mask = series > final_thresh

        else:  # "below"
            candidates = series[series < base_thresh]
            if candidates.empty:
                results[metric] = {"windows": [], "values": [], "threshold": base_thresh, "direction": direction}
                continue

            # Among those, keep only the bottom `max_fraction` worst (lowest) values
            dyn_thresh = candidates.quantile(max_fraction)
            final_thresh = min(base_thresh, dyn_thresh)
            mask = series < final_thresh

        bad_rows = df.loc[mask]

        results[metric] = {
            "windows": bad_rows["time_window"].tolist(),
            "values":  bad_rows[metric].tolist(),
            "threshold": final_thresh,
            "direction": direction,
        }

    return results

def format_metric_feedback(violations, window_seconds=10):
    texts = []
    readable = {
        "shoulder_slope_mean": "shoulder alignment was off",
        "spine_angle_mean": "your posture showed slouching or leaning",
        "eye_openness_mean": "your eye openness was low (possible disengagement)",
        "mouth_gap_mean": "your mouth tension/movement increased (possible nervousness)",
    }

    for metric, data in violations.items():
        windows = data["windows"]
        if not windows:
            continue

        times = []
        for w in windows:
            start = w * window_seconds
            end = start + window_seconds
            times.append(f"{int(start)}s–{int(end)}s")

        time_str = ", ".join(times)
        texts.append(f"- Between {time_str}, {readable[metric]}.")

    return "\n".join(texts)

# ========== MAIN ENTRY POINT ==========
def main():
    parser = argparse.ArgumentParser(description="Behavior analysis from interview video.")
    parser.add_argument("video_path", type=str, help="Path to the video file")

    args = parser.parse_args()
    video_path = args.video_path

    if not os.path.exists(video_path):
        print(f"File not found: {video_path}")
        sys.exit(1)

    summary = summarize_results(video_path, window_seconds=10)

    if summary.empty:
        print("No usable data from video.")
        sys.exit(0)

    violations = detect_behavior_violations(summary, max_fraction=0.3)
    feedback = format_metric_feedback(violations, window_seconds=10)


    print("\n===== Behavioral Feedback =====\n")
    print(feedback if feedback else "No significant negative behavioral changes detected.")
    print("\n================================\n")


if __name__ == "__main__":
    main()
