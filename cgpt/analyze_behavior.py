#!/usr/bin/env python3

import os
import sys
import argparse
import json
import pickle
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
import eyed3
from moviepy.editor import VideoFileClip
from pyAudioAnalysis import audioBasicIO, ShortTermFeatures
import librosa
try:
    import xgboost as xgb
except ImportError:
    xgb = None


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

    try:
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

        return pd.DataFrame(rows)
    finally:
        cap.release()
        # MediaPipe solutions will be garbage collected automatically


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


def summarize_results_for_model(video_path):
    audio_path = extract_audio(video_path)
    audio_features = analyze_audio(audio_path)
    video_features = analyze_video(video_path)

    if video_features.empty:
        os.remove(audio_path)
        return pd.DataFrame()

    # aggregate per video instead of per minute
    summary = video_features.agg({
        "shoulder_slope": ["mean", "std"],
        "spine_angle": ["mean", "std"],
        "eye_openness": ["mean", "std"],
        "mouth_gap": ["mean", "std"]
    }).T.unstack().to_frame().T  # single row

    summary.columns = [f"{col[0]}_{col[1]}" for col in summary.columns]
    for k, v in audio_features.items():
        summary[k] = v

    os.remove(audio_path)
    return summary


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


def format_model_predictions(preds_df):
    """
    Format model predictions into human-readable text.
    
    Args:
        preds_df: DataFrame with prediction columns (engaged_pred, engagingtone_pred, etc.)
    
    Returns:
        Formatted string with model predictions
    """
    if preds_df.empty:
        return ""
    
    metric_names = {
        "engaged_pred": "Engaged",
        "engagingtone_pred": "Engaging Tone",
        "excited_pred": "Excitement",
        "friendly_pred": "Friendliness",
        "smile_pred": "Smile"
    }
    
    texts = []
    
    for col in preds_df.columns:
        metric_name = metric_names.get(col, col.replace("_pred", "").title())
        value = preds_df[col].iloc[0] if len(preds_df) > 0 else 0.0
        # Format as raw score value
        texts.append(f"{metric_name}: {value:.3f}")
    
    return "\n".join(texts)


def load_bundle(model_dir):
    """
    Load model, scaler and feature_names for one metric.
    """
    if xgb is None:
        raise ImportError("xgboost is not installed. Please install it with: pip install xgboost")
    
    model_path = os.path.join(model_dir, "xgb_model.json")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    feature_names_path = os.path.join(model_dir, "feature_names.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    if not os.path.exists(feature_names_path):
        raise FileNotFoundError(f"Feature names file not found: {feature_names_path}")

    # model
    model = xgb.XGBRegressor()
    model.load_model(model_path)

    # scaler
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # feature names (order matters!)
    with open(feature_names_path, "r") as f:
        feature_names = json.load(f)

    return model, scaler, feature_names


def score_all_metrics(df_input, models_root):
    """
    df_input: pandas DataFrame with all available features (80–90 cols).
    models_root: directory containing subfolders for each metric.

    Returns: DataFrame with 5 prediction columns.
    """
    metrics = ["engaged", "engagingtone", "excited", "friendly", "smile"]
    preds_dict = {}

    for metric in metrics:
        metric_dir = os.path.join(models_root, metric)
        
        if not os.path.exists(metric_dir):
            print(f"⚠️ Warning: Model directory not found for '{metric}'. Skipping.")
            preds_dict[f"{metric}_pred"] = [0.0] * len(df_input)
            continue
        
        try:
            model, scaler, feature_names = load_bundle(metric_dir)

            # check that required features exist
            missing = [c for c in feature_names if c not in df_input.columns]
            if missing:
                print(f"⚠️ Warning: [{metric}] missing required features: {missing}. Skipping.")
                preds_dict[f"{metric}_pred"] = [0.0] * len(df_input)
                continue

            # select and order columns as during training
            X = df_input[feature_names].copy()

            # scale
            X_scaled = scaler.transform(X)

            # predict
            y_pred = model.predict(X_scaled)
            
            # Handle single value vs array
            if isinstance(y_pred, np.ndarray) and len(y_pred) > 0:
                preds_dict[f"{metric}_pred"] = [float(y_pred[0])]
            else:
                preds_dict[f"{metric}_pred"] = [float(y_pred)]

        except Exception as e:
            print(f"⚠️ Warning: Error predicting {metric}: {str(e)}")
            preds_dict[f"{metric}_pred"] = [0.0] * len(df_input)

    # return predictions as a DataFrame (index aligned with input df)
    preds_df = pd.DataFrame(preds_dict, index=df_input.index)
    return preds_df


# ========== MAIN ENTRY POINT ==========
def main():
    parser = argparse.ArgumentParser(description="Behavior analysis from interview video.")
    parser.add_argument("video_path", type=str, help="Path to the video file")

    args = parser.parse_args()
    video_path = args.video_path

    if not os.path.exists(video_path):
        print(f"File not found: {video_path}")
        sys.exit(1)

    # Descriptive analysis (window-based)
    summary = summarize_results(video_path, window_seconds=10)

    if summary.empty:
        print("No usable data from video.")
        sys.exit(0)

    violations = detect_behavior_violations(summary, max_fraction=0.3)
    feedback = format_metric_feedback(violations, window_seconds=10)

    print(feedback if feedback else "No significant negative behavioral changes detected.")

    # Model-based predictions (aggregate features per video)
    try:
        model_summary = summarize_results_for_model(video_path)
        
        if model_summary.empty:
            print("\n⚠️ Warning: Could not extract features for model predictions.")
        else:
            models_root = "vision_models"
            
            if not os.path.exists(models_root):
                print(f"\n⚠️ Warning: Models directory not found at '{models_root}'. Skipping model predictions.")
            else:
                try:
                    preds_df = score_all_metrics(model_summary, models_root)
                    
                    metric_names = {
                        "engaged_pred": "Engaged",
                        "engagingtone_pred": "Engaging Tone",
                        "excited_pred": "Excitement",
                        "friendly_pred": "Friendliness",
                        "smile_pred": "Smile"
                    }
                    
                    for col in preds_df.columns:
                        metric_name = metric_names.get(col, col.replace("_pred", "").title())
                        value = preds_df[col].iloc[0] if len(preds_df) > 0 else 0.0
                        # Format as percentage or score (assuming 0-1 range)
                        print(f"{metric_name}: {value:.3f}")
                    
                except Exception as e:
                    print(f"\n⚠️ Warning: Error running model predictions: {str(e)}")
                    import traceback
                    traceback.print_exc()
    
    except Exception as e:
        print(f"\n⚠️ Warning: Error extracting features for model predictions: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
