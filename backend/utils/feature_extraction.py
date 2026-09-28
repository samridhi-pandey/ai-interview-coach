"""Feature Extraction Utilities for Multimodal Emotion & Intent Analyzer.

This module provides preprocessing and feature extraction functions for
text, audio, and video inputs. It includes deterministic signal processing
for audio prosody and facial geometry calculations for webcam frames.
"""

import math
from typing import Dict, List
import numpy as np


def preprocess_text(text: str) -> Dict:
    """Preprocesses text and extracts basic linguistic statistics.

    Args:
        text: The raw input string to analyze.

    Returns:
        A dictionary containing the cleaned text, word count, character count,
        exclamation/question counts, capital letter ratio, and average word length.
    """
    tokens = text.lower().split()
    char_count = len(text)
    word_count = len(tokens)
    return {
        "text": text,
        "word_count": word_count,
        "char_count": char_count,
        "exclamation_count": text.count("!"),
        "question_count": text.count("?"),
        "capital_ratio": round(sum(1 for c in text if c.isupper()) / max(char_count, 1), 4),
        "avg_word_len": round(sum(len(t) for t in tokens) / max(word_count, 1), 2),
    }


def compute_rms_energy(samples: List[float]) -> float:
    """Computes the Root Mean Square (RMS) energy of a frame of audio samples.

    Args:
        samples: A list or numpy array of audio samples.

    Returns:
        The RMS energy value as a float.
    """
    if len(samples) == 0:
        return 0.0
    return math.sqrt(sum(x**2 for x in samples) / len(samples))


def compute_zero_crossing_rate(samples: List[float]) -> float:
    """Computes the Zero-Crossing Rate (ZCR) of a frame of audio samples.

    Args:
        samples: A list or numpy array of audio samples.

    Returns:
        The ratio of zero crossings to the total number of samples.
    """
    if len(samples) < 2:
        return 0.0
    crossings = sum(
        1 for i in range(1, len(samples))
        if (samples[i] >= 0) != (samples[i-1] >= 0)
    )
    return crossings / len(samples)


def estimate_pitch(samples: np.ndarray, sample_rate: int = 16000) -> float:
    """Estimates the fundamental frequency (pitch) using autocorrelation.

    Args:
        samples: A numpy array of raw audio samples.
        sample_rate: The audio sample rate in Hz.

    Returns:
        The estimated pitch frequency in Hz, or 0.0 if not detectable.
    """
    if len(samples) < 512:
        return 0.0

    # Center-clip or mean-center the signal
    samples_centered = samples - np.mean(samples)
    
    # Compute autocorrelation
    corr = np.correlate(samples_centered, samples_centered, mode='full')
    corr = corr[len(corr)//2:]
    
    # Look for pitch period in human vocal range (80 Hz to 400 Hz)
    min_lag = int(sample_rate / 400)
    max_lag = int(sample_rate / 80)
    
    if min_lag >= len(corr) or max_lag >= len(corr):
        return 0.0
        
    peak = np.argmax(corr[min_lag:max_lag]) + min_lag
    pitch = sample_rate / peak if peak > 0 else 0.0
    return pitch


def trim_silence(samples: np.ndarray, threshold: float = 0.01, frame_size: int = 512) -> np.ndarray:
    """Trims leading and trailing silence based on frame-level RMS energy.

    Args:
        samples: A numpy array of raw audio samples.
        threshold: The RMS energy threshold below which a frame is silent.
        frame_size: Number of samples per frame.

    Returns:
        A new numpy array containing only active audio frames.
    """
    if len(samples) < frame_size:
        return samples
    
    num_frames = len(samples) // frame_size
    active_frames = []
    
    for i in range(num_frames):
        frame = samples[i*frame_size:(i+1)*frame_size]
        rms = np.sqrt(np.mean(frame**2))
        if rms > threshold:
            active_frames.append(frame)
            
    if not active_frames:
        return samples
        
    return np.concatenate(active_frames)


def extract_prosody_features(samples: List[float], sample_rate: int = 16000,
                             frame_size: int = 512, hop_size: int = 256) -> Dict:
    """Extracts raw prosody metrics from a sequence of audio samples.

    Args:
        samples: List or array of raw audio samples.
        sample_rate: Sample rate of the audio in Hz.
        frame_size: Frame size for analysis.
        hop_size: Step size between frames.

    Returns:
        A dictionary containing energy mean/std, ZCR mean, duration, and pitch metrics.
    """
    samples_arr = np.array(samples, dtype=np.float32)
    
    # 1. Deterministic preprocessing: normalize amplitude to [-1.0, 1.0]
    max_val = np.max(np.abs(samples_arr))
    if max_val > 0:
        samples_arr /= max_val
        
    # 2. Trim silence (VAD-like preprocessing)
    samples_arr = trim_silence(samples_arr, threshold=0.015, frame_size=frame_size)
    
    # Scale hop size dynamically for long audio (maintaining ~25-50 fps for speech prosody)
    actual_hop = hop_size
    if len(samples_arr) > sample_rate * 20:
        actual_hop = max(hop_size, int(sample_rate * 0.04)) # 25 frames/sec for extended audio

    frames = []
    for start in range(0, len(samples_arr) - frame_size + 1, actual_hop):
        frames.append(samples_arr[start:start + frame_size])

    if not frames:
        # Fallback to untrimmed audio and pad with zeros so prosody is never dropped
        samples_arr = np.array(samples, dtype=np.float32)
        if len(samples_arr) == 0:
            return {"error": "audio signal is empty"}
        max_val = np.max(np.abs(samples_arr))
        if max_val > 0:
            samples_arr /= max_val
        if len(samples_arr) < frame_size:
            samples_arr = np.pad(samples_arr, (0, frame_size - len(samples_arr)))
        for start in range(0, len(samples_arr) - frame_size + 1, actual_hop):
            frames.append(samples_arr[start:start + frame_size])
        if not frames:
            frames = [samples_arr[:frame_size]]

    energies = [compute_rms_energy(f) for f in frames]
    zcrs     = [compute_zero_crossing_rate(f) for f in frames]
    
    # Pitch extraction: evaluate across voiced frames (human pitch requires vocal fold vibration)
    # Subsample up to 250 evenly spaced voiced frames across the full timeline for high precision without lag
    voiced_frames = [f for i, f in enumerate(frames) if energies[i] > 0.02]
    if not voiced_frames:
        voiced_frames = frames

    if len(voiced_frames) > 250:
        step = len(voiced_frames) / 250.0
        pitch_eval_frames = [voiced_frames[int(k * step)] for k in range(250)]
    else:
        pitch_eval_frames = voiced_frames

    pitches = []
    for f in pitch_eval_frames:
        p = estimate_pitch(f, sample_rate)
        if p > 0:
            pitches.append(p)
            
    pitch_mean = float(np.mean(pitches)) if pitches else 120.0
    pitch_std = float(np.std(pitches)) if pitches else 15.0

    return {
        "energy_mean":    round(float(np.mean(energies)), 4),
        "energy_std":     round(float(np.std(energies)),  4),
        "zcr_mean":       round(float(np.mean(zcrs)),     4),
        "pitch_mean_hz":  round(pitch_mean, 1),
        "pitch_std_hz":   round(pitch_std, 1),
        "duration_s":     round(len(samples_arr) / sample_rate, 3),
        "frame_count":    len(frames),
    }


def calculate_ear(landmarks: List[Dict]) -> float:
    """Computes the Eye Aspect Ratio (EAR) given 3D facial landmarks.

    Args:
        landmarks: List of landmark points (dicts with 'x', 'y', 'z').

    Returns:
        The average Eye Aspect Ratio as a float.
    """
    if len(landmarks) < 468:
        return 0.3
    
    # Left eye vertical: 385-380, 386-374, 387-373. Horizontal: 362-263
    p385 = landmarks[385]
    p380 = landmarks[380]
    p386 = landmarks[386]
    p374 = landmarks[374]
    p387 = landmarks[387]
    p373 = landmarks[373]
    p362 = landmarks[362]
    p263 = landmarks[263]
    
    d_v1 = math.sqrt((p385['x']-p380['x'])**2 + (p385['y']-p380['y'])**2)
    d_v2 = math.sqrt((p386['x']-p374['x'])**2 + (p386['y']-p374['y'])**2)
    d_v3 = math.sqrt((p387['x']-p373['x'])**2 + (p387['y']-p373['y'])**2)
    d_h = math.sqrt((p362['x']-p263['x'])**2 + (p362['y']-p263['y'])**2)
    ear_left = (d_v1 + d_v2 + d_v3) / (2.0 * max(d_h, 1e-6))
    
    # Right eye vertical: 158-153, 159-145, 160-144. Horizontal: 33-133
    p158 = landmarks[158]
    p153 = landmarks[153]
    p159 = landmarks[159]
    p145 = landmarks[145]
    p160 = landmarks[160]
    p144 = landmarks[144]
    p33 = landmarks[33]
    p133 = landmarks[133]
    
    d_v1_r = math.sqrt((p158['x']-p153['x'])**2 + (p158['y']-p153['y'])**2)
    d_v2_r = math.sqrt((p159['x']-p145['x'])**2 + (p159['y']-p145['y'])**2)
    d_v3_r = math.sqrt((p160['x']-p144['x'])**2 + (p160['y']-p144['y'])**2)
    d_h_r = math.sqrt((p33['x']-p133['x'])**2 + (p33['y']-p133['y'])**2)
    ear_right = (d_v1_r + d_v2_r + d_v3_r) / (2.0 * max(d_h_r, 1e-6))
    
    return (ear_left + ear_right) / 2.0


def calculate_mar(landmarks: List[Dict]) -> float:
    """Computes the Mouth Aspect Ratio (MAR) given 3D facial landmarks.

    Args:
        landmarks: List of landmark points (dicts with 'x', 'y', 'z').

    Returns:
        The Mouth Aspect Ratio as a float.
    """
    if len(landmarks) < 468:
        return 0.2
    
    # Mouth vertical: 13-14 (inner lips). Horizontal: 78-308
    p13 = landmarks[13]
    p14 = landmarks[14]
    p78 = landmarks[78]
    p308 = landmarks[308]
    
    d_v = math.sqrt((p13['x']-p14['x'])**2 + (p13['y']-p14['y'])**2)
    d_h = math.sqrt((p78['x']-p308['x'])**2 + (p78['y']-p308['y'])**2)
    
    return d_v / max(d_h, 1e-6)


def calculate_brow_distance(landmarks: List[Dict]) -> float:
    """Computes the vertical distance between eyebrows and eyes normalized by eye width.

    Args:
        landmarks: List of landmark points (dicts with 'x', 'y', 'z').

    Returns:
        Average eyebrow-to-eye vertical distance normalized by outer eye width as a float.
    """
    if len(landmarks) < 468:
        return 0.35
    p70 = landmarks[70]
    p159 = landmarks[159]
    p300 = landmarks[300]
    p386 = landmarks[386]
    p33 = landmarks[33]
    p263 = landmarks[263]
    
    eye_width = math.sqrt((p33['x']-p263['x'])**2 + (p33['y']-p263['y'])**2)
    dist_l = math.sqrt((p70['x']-p159['x'])**2 + (p70['y']-p159['y'])**2)
    dist_r = math.sqrt((p300['x']-p386['x'])**2 + (p300['y']-p386['y'])**2)
    return ((dist_l + dist_r) / 2.0) / max(eye_width, 1e-6)


def calculate_smile_ratio(landmarks: List[Dict]) -> float:
    """Computes the Lip Corner Puller (AU12) smile ratio.

    Normalizes mouth width by interpupillary / outer eye distance.

    Args:
        landmarks: List of landmark points (dicts with 'x', 'y', 'z').

    Returns:
        Normalized smile ratio as a float (typically 0.42-0.65).
    """
    if len(landmarks) < 468:
        return 0.45
    p33 = landmarks[33]
    p263 = landmarks[263]
    p61 = landmarks[61]
    p291 = landmarks[291]
    
    eye_width = math.sqrt((p33['x']-p263['x'])**2 + (p33['y']-p263['y'])**2)
    mouth_width = math.sqrt((p61['x']-p291['x'])**2 + (p61['y']-p291['y'])**2)
    
    return mouth_width / max(eye_width, 1e-6)


def calculate_brow_furrow(landmarks: List[Dict]) -> float:
    """Computes the Brow Lowerer / Corrugator (AU4) inner eyebrow distance.

    Narrows when the speaker is angry, intensely concentrating, or frustrated.

    Args:
        landmarks: List of landmark points (dicts with 'x', 'y', 'z').

    Returns:
        Normalized inner brow distance as a float.
    """
    if len(landmarks) < 468:
        return 0.20
    p107 = landmarks[107]
    p336 = landmarks[336]
    p33 = landmarks[33]
    p263 = landmarks[263]
    
    eye_width = math.sqrt((p33['x']-p263['x'])**2 + (p33['y']-p263['y'])**2)
    brow_gap = math.sqrt((p107['x']-p336['x'])**2 + (p107['y']-p336['y'])**2)
    
    return brow_gap / max(eye_width, 1e-6)


def calculate_smirk_asymmetry(landmarks: List[Dict]) -> float:
    """Computes unilateral lip corner asymmetry (AU14 / AU12 unilateral).

    Detects asymmetric smiles, sneers, and contemptuous smirks in face-aligned coordinates.

    Args:
        landmarks: List of landmark points (dicts with 'x', 'y', 'z').

    Returns:
        Asymmetry delta between left and right lip corners as a float.
    """
    if len(landmarks) < 468:
        return 0.0
    p61 = landmarks[61]
    p291 = landmarks[291]
    p33 = landmarks[33]
    p263 = landmarks[263]
    
    # Compute face-aligned axis from outer eye corners to eliminate head-tilt artifact
    dx = p263['x'] - p33['x']
    dy = p263['y'] - p33['y']
    eye_len = math.sqrt(dx * dx + dy * dy)
    if eye_len < 1e-6:
        return 0.0
    ux, uy = dx / eye_len, dy / eye_len
    vx, vy = -uy, ux  # perpendicular axis pointing down face midline

    proj_l = (p61['x'] - p33['x']) * vx + (p61['y'] - p33['y']) * vy
    proj_r = (p291['x'] - p263['x']) * vx + (p291['y'] - p263['y']) * vy
    return abs(proj_l - proj_r) / eye_len


def extract_video_features(landmarks: List[Dict]) -> Dict:
    """Extracts facial Action Unit geometry features from MediaPipe landmarks.

    Args:
        landmarks: List of landmark points (dicts with 'x', 'y', 'z').

    Returns:
        A dictionary containing EAR, MAR, brow distance, smile ratio, brow furrow,
        smirk asymmetry, blink status, mouth open status, and smiling detection.
    """
    ear = calculate_ear(landmarks)
    mar = calculate_mar(landmarks)
    brow_dist = calculate_brow_distance(landmarks)
    smile_ratio = calculate_smile_ratio(landmarks)
    brow_furrow = calculate_brow_furrow(landmarks)
    smirk_asym = calculate_smirk_asymmetry(landmarks)
    
    return {
        "ear": round(ear, 3),
        "mar": round(mar, 3),
        "brow_dist": round(brow_dist, 3),
        "smile_ratio": round(smile_ratio, 3),
        "brow_furrow": round(brow_furrow, 3),
        "smirk_asymmetry": round(smirk_asym, 3),
        "blink_detected": ear < 0.22,
        "mouth_open": mar > 0.45,
        "smiling": smile_ratio > 0.52
    }