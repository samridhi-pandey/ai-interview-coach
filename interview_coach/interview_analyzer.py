"""Interview Preparation AI Coach - Facial & Vocal Analysis Engine.

This module delivers a self-contained, high-performance facial and vocal analysis pipeline
for candidate interview practice. It ingests video/audio streams, extracts MediaPipe FaceMesh
Action Units and audio DSP prosody, runs local Whisper ASR, detects filler words and pauses,
and synthesizes actionable coaching rubrics on a 0-100 scale.
"""

import base64
import io
import math
import os
import re
import subprocess
import sys
import tempfile
import wave
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import imageio_ffmpeg
import numpy as np

# Ensure parent workspaces are resolvable for modular imports
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_CURRENT_DIR)
for _p in [_CURRENT_DIR, _PARENT_DIR, r"C:\tcs", r"C:\multimodel-emotionAI"]:
    if os.path.exists(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

# Import feature extraction functions (self-contained within package or workspace)
try:
    from utils.feature_extraction import (
        calculate_brow_distance,
        calculate_brow_furrow,
        calculate_ear,
        calculate_mar,
        calculate_smile_ratio,
        calculate_smirk_asymmetry,
        compute_rms_energy,
        compute_zero_crossing_rate,
        estimate_pitch,
        extract_prosody_features,
        extract_video_features,
        trim_silence,
    )
except ImportError:
    try:
        from interview_coach.utils.feature_extraction import (
            calculate_brow_distance,
            calculate_brow_furrow,
            calculate_ear,
            calculate_mar,
            calculate_smile_ratio,
            calculate_smirk_asymmetry,
            compute_rms_energy,
            compute_zero_crossing_rate,
            estimate_pitch,
            extract_prosody_features,
            extract_video_features,
            trim_silence,
        )
    except ImportError:
        from backend.utils.feature_extraction import (
            calculate_brow_distance,
            calculate_brow_furrow,
            calculate_ear,
            calculate_mar,
            calculate_smile_ratio,
            calculate_smirk_asymmetry,
            compute_rms_energy,
            compute_zero_crossing_rate,
            estimate_pitch,
            extract_prosody_features,
            extract_video_features,
            trim_silence,
        )

# MediaPipe FaceMesh initialization
import mediapipe as mp

# Lazy-loaded globals for neural pipeline
_ASR_PIPELINE = None
_MP_FACE_MESH = None

# Regex pattern for interview filler words
FILLER_REGEX = re.compile(
    r"\b(um|uh|er|ah|like|you\s+know|sort\s+of|kind\s+of|actually|basically|literally|i\s+mean)\b",
    re.IGNORECASE,
)


def get_face_mesh_detector():
    """Initializes or returns singleton MediaPipe FaceMesh instance."""
    global _MP_FACE_MESH
    if _MP_FACE_MESH is None:
        try:
            mp_face_mesh = mp.solutions.face_mesh
            _MP_FACE_MESH = mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
            )
        except Exception as e:
            print(f"[InterviewAnalyzer] Warning: MediaPipe FaceMesh init fallback: {e}")
            _MP_FACE_MESH = False
    return _MP_FACE_MESH if _MP_FACE_MESH is not False else None


def get_asr_pipeline():
    """Returns local cached Whisper ASR pipeline for fast speech-to-text."""
    global _ASR_PIPELINE
    if _ASR_PIPELINE is not None and _ASR_PIPELINE is not False:
        return _ASR_PIPELINE

    try:
        import torch
        from transformers import pipeline

        asr_device = 0 if torch.cuda.is_available() else -1

        # Prefer tiny.en for low latency (< 1s execution)
        for model_id in ["openai/whisper-tiny.en", "openai/whisper-base.en"]:
            try:
                _ASR_PIPELINE = pipeline(
                    "automatic-speech-recognition",
                    model=model_id,
                    device=asr_device,
                )
                print(f"[InterviewAnalyzer] Loaded Whisper pipeline: {model_id}")
                break
            except Exception as load_err:
                print(f"[InterviewAnalyzer] Attempt to load {model_id}: {load_err}")

    except Exception as e:
        print(f"[InterviewAnalyzer] Warning: ASR pipeline unavailable: {e}")
        _ASR_PIPELINE = False

    return _ASR_PIPELINE if _ASR_PIPELINE is not False else None


@dataclass
class FacialMetrics:
    """Detailed facial expressiveness and poise metrics."""

    frames_analyzed: int = 0
    faces_detected_count: int = 0
    eye_contact_ratio: float = 0.0  # Percentage [0-100]
    looking_away_ratio: float = 0.0  # Percentage [0-100]
    blink_count: int = 0
    blink_rate_bpm: float = 0.0  # Blinks per minute
    ear_stability: float = 0.0
    smile_ratio_mean: float = 0.0
    smiling_ratio: float = 0.0  # Percentage [0-100]
    brow_furrow_mean: float = 0.0
    brow_tension_ratio: float = 0.0  # Percentage [0-100]
    facial_expressiveness: float = 0.0  # Score [0-100]
    head_stability_score: float = 0.0  # Score [0-100]
    warmth_score: float = 0.0  # Score [0-100]


@dataclass
class VocalMetrics:
    """Detailed vocal prosody and delivery metrics."""

    rms_energy_mean: float = 0.0
    rms_energy_std: float = 0.0
    vocal_energy_score: float = 0.0  # Score [0-100]
    pitch_mean_hz: float = 0.0
    pitch_std_hz: float = 0.0
    pitch_dynamic_score: float = 0.0  # Score [0-100]
    is_monotone: bool = False
    words_per_minute: float = 0.0
    pacing_score: float = 0.0  # Score [0-100]
    pause_ratio: float = 0.0  # Ratio [0.0 - 1.0]
    speech_duration_s: float = 0.0


@dataclass
class SpeechMetrics:
    """Transcript, word count, and filler word detection."""

    transcript: str = ""
    word_count: int = 0
    filler_words_count: int = 0
    filler_words_detected: Dict[str, int] = field(default_factory=dict)
    filler_word_ratio: float = 0.0  # Percentage [0-100]


@dataclass
class AnalysisResult:
    """Consolidated interview coaching result."""

    overall_delivery_score: float = 0.0
    confidence_score: float = 0.0
    clarity_score: float = 0.0
    engagement_score: float = 0.0
    scores: Dict[str, float] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    strengths: List[str] = field(default_factory=list)
    actionable_improvements: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Converts result object to clean JSON-serializable dictionary."""
        return {
            "success": True,
            "overall_delivery_score": round(self.overall_delivery_score, 1),
            "confidence_score": round(self.confidence_score, 1),
            "clarity_score": round(self.clarity_score, 1),
            "engagement_score": round(self.engagement_score, 1),
            "scores": {
                "overall_delivery_score": round(self.overall_delivery_score, 1),
                "confidence_score": round(self.confidence_score, 1),
                "clarity_score": round(self.clarity_score, 1),
                "engagement_score": round(self.engagement_score, 1),
            },
            "metrics": self.metrics,
            "strengths": self.strengths,
            "actionable_improvements": self.actionable_improvements,
        }


class InterviewAnalyzer:
    """Core analysis engine for interview performance coaching."""

    def __init__(self):
        """Initializes the interview analyzer."""
        self.face_mesh = get_face_mesh_detector()
        self.ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    # -------------------------------------------------------------------------
    # Audio Separation & Parsing
    # -------------------------------------------------------------------------
    def extract_audio_from_video(self, video_path: str) -> Tuple[Optional[np.ndarray], int, float]:
        """Extracts 16kHz mono WAV audio samples from a video file using imageio-ffmpeg.

        Args:
            video_path: Path to the input video file.

        Returns:
            Tuple of (samples array, sample_rate, duration_in_seconds).
        """
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            out_wav = tf.name

        try:
            cmd = [
                self.ffmpeg_exe,
                "-y",
                "-i",
                video_path,
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-f",
                "wav",
                out_wav,
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res.returncode == 0 and os.path.exists(out_wav) and os.path.getsize(out_wav) > 44:
                return self.parse_wav_file(out_wav)
        except Exception as e:
            print(f"[InterviewAnalyzer] Audio extraction warning: {e}")
        finally:
            if os.path.exists(out_wav):
                try:
                    os.remove(out_wav)
                except Exception:
                    pass

        return None, 16000, 0.0

    def parse_wav_file(self, wav_path: str) -> Tuple[Optional[np.ndarray], int, float]:
        """Parses a WAV file on disk into normalized float32 samples.

        Args:
            wav_path: Local path to the WAV file.

        Returns:
            Tuple of (samples array, sample_rate, duration_in_seconds).
        """
        try:
            with wave.open(wav_path, "rb") as wf:
                sr = wf.getframerate()
                n_frames = wf.getnframes()
                sampwidth = wf.getsampwidth()
                n_channels = wf.getnchannels()
                raw_data = wf.readframes(n_frames)

            if sampwidth == 2:
                samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
            elif sampwidth == 1:
                samples = (np.frombuffer(raw_data, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
            elif sampwidth == 4:
                samples = np.frombuffer(raw_data, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0

            if n_channels > 1:
                samples = samples.reshape(-1, n_channels).mean(axis=1)

            duration_s = len(samples) / float(sr) if sr > 0 else 0.0
            return samples, sr, duration_s
        except Exception as e:
            print(f"[InterviewAnalyzer] WAV parse error: {e}")
            return None, 16000, 0.0

    def parse_wav_bytes(self, wav_bytes: bytes) -> Tuple[Optional[np.ndarray], int, float]:
        """Parses in-memory audio bytes into 16kHz float32 audio samples.

        Supports standard WAV and uses ffmpeg transcode fallback for WebM/MP3/AAC.

        Args:
            wav_bytes: Raw audio byte buffer.

        Returns:
            Tuple of (samples array, sample_rate, duration_in_seconds).
        """
        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
                sr = wf.getframerate()
                n_frames = wf.getnframes()
                sampwidth = wf.getsampwidth()
                n_channels = wf.getnchannels()
                raw_data = wf.readframes(n_frames)

            if sampwidth == 2:
                samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
            elif sampwidth == 1:
                samples = (np.frombuffer(raw_data, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
            elif sampwidth == 4:
                samples = np.frombuffer(raw_data, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0

            if n_channels > 1:
                samples = samples.reshape(-1, n_channels).mean(axis=1)

            duration_s = len(samples) / float(sr) if sr > 0 else 0.0
            return samples, sr, duration_s
        except Exception:
            # Fallback through ffmpeg stream pipe
            with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as in_f:
                in_f.write(wav_bytes)
                in_name = in_f.name
            out_name = in_name + ".wav"

            try:
                cmd = [
                    self.ffmpeg_exe,
                    "-y",
                    "-i",
                    in_name,
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-f",
                    "wav",
                    out_name,
                ]
                subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if os.path.exists(out_name) and os.path.getsize(out_name) > 44:
                    return self.parse_wav_file(out_name)
            finally:
                for fn in [in_name, out_name]:
                    if os.path.exists(fn):
                        try:
                            os.remove(fn)
                        except Exception:
                            pass

        return None, 16000, 0.0

    # -------------------------------------------------------------------------
    # Video Frame Sampling
    # -------------------------------------------------------------------------
    def sample_video_frames(
        self, video_path: str, max_samples: int = 30
    ) -> Tuple[List[np.ndarray], float]:
        """Evenly samples video frames across the timeline for facial analysis.

        Args:
            video_path: Path to the video file.
            max_samples: Target maximum number of frames to sample.

        Returns:
            Tuple of (list of BGR numpy frame images, video duration in seconds).
        """
        frames: List[np.ndarray] = []
        duration_s = 0.0

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return frames, duration_s

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_s = (total_frames / fps) if fps > 0 and total_frames > 0 else 0.0

        if total_frames > 0:
            sample_count = min(max_samples, total_frames)
            # Generate evenly spaced frame indices avoiding edge transients
            indices = np.linspace(
                max(0, int(total_frames * 0.03)),
                min(total_frames - 1, int(total_frames * 0.97)),
                num=sample_count,
                dtype=int,
            )
            # Remove duplicates while preserving ordering
            indices = sorted(list(set(indices)))

            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if ret and frame is not None:
                    frames.append(frame)

        cap.release()
        return frames, duration_s

    # -------------------------------------------------------------------------
    # 1. Facial Analysis (MediaPipe FaceMesh)
    # -------------------------------------------------------------------------
    def analyze_facial_frames(
        self, frames: List[np.ndarray], duration_s: float = 0.0
    ) -> FacialMetrics:
        """Analyzes facial Action Units, eye contact, blinks, warmth, and poise across frames.

        Args:
            frames: List of BGR video frame arrays.
            duration_s: Estimated video duration in seconds.

        Returns:
            FacialMetrics dataclass populated with computed metrics.
        """
        if not frames:
            return FacialMetrics()

        detector = get_face_mesh_detector()
        if not detector:
            return FacialMetrics(frames_analyzed=len(frames))

        ears: List[float] = []
        mars: List[float] = []
        smile_ratios: List[float] = []
        brow_furrows: List[float] = []
        nose_positions: List[Tuple[float, float]] = []
        eye_contact_flags: List[bool] = []
        looking_away_flags: List[bool] = []
        blink_flags: List[bool] = []

        faces_detected = 0

        for frame in frames:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.process(rgb_frame)

            if not results.multi_face_landmarks:
                looking_away_flags.append(True)
                eye_contact_flags.append(False)
                continue

            faces_detected += 1
            face_lms = results.multi_face_landmarks[0]
            landmarks = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in face_lms.landmark]

            # Action Unit extraction via feature_extraction utilities
            ear = calculate_ear(landmarks)
            mar = calculate_mar(landmarks)
            smile = calculate_smile_ratio(landmarks)
            furrow = calculate_brow_furrow(landmarks)

            ears.append(ear)
            mars.append(mar)
            smile_ratios.append(smile)
            brow_furrows.append(furrow)

            # Blink detection (EAR drops below 0.21)
            is_blink = ear < 0.21
            blink_flags.append(is_blink)

            # Head orientation & Eye Contact geometry
            # Landmark 33: outer left eye corner, 263: outer right eye corner, 1: nose tip
            # Landmark 10: forehead center, 152: chin
            p33 = landmarks[33]
            p263 = landmarks[263]
            p1 = landmarks[1]
            p10 = landmarks[10]
            p152 = landmarks[152]

            eye_center_x = (p33["x"] + p263["x"]) / 2.0
            eye_span = max(abs(p263["x"] - p33["x"]), 1e-5)
            yaw_offset = (p1["x"] - eye_center_x) / eye_span  # Normal ~0.0

            eye_center_y = (p33["y"] + p263["y"]) / 2.0
            face_height = max(abs(p152["y"] - p10["y"]), 1e-5)
            pitch_rel = (p1["y"] - eye_center_y) / face_height  # Normal ~0.18-0.38

            nose_positions.append((p1["x"], p1["y"]))

            # Eye contact: head facing camera directly, pitch level, and eyes open
            has_eye_contact = (
                abs(yaw_offset) <= 0.18
                and 0.10 <= pitch_rel <= 0.42
                and ear >= 0.19
            )
            is_looking_away = abs(yaw_offset) > 0.22 or pitch_rel < 0.08 or pitch_rel > 0.46

            eye_contact_flags.append(has_eye_contact)
            looking_away_flags.append(is_looking_away)

        total_frames = len(frames)
        if faces_detected == 0:
            return FacialMetrics(
                frames_analyzed=total_frames,
                faces_detected_count=0,
                eye_contact_ratio=0.0,
                looking_away_ratio=100.0,
                warmth_score=45.0,
                head_stability_score=50.0,
            )

        # 1. Eye Contact & Blink Rate
        eye_contact_ratio = round((sum(eye_contact_flags) / max(total_frames, 1)) * 100.0, 1)
        looking_away_ratio = round((sum(looking_away_flags) / max(total_frames, 1)) * 100.0, 1)

        # Count blink transitions (open -> closed -> open)
        blink_count = 0
        in_blink = False
        for b in blink_flags:
            if b and not in_blink:
                blink_count += 1
                in_blink = True
            elif not b:
                in_blink = False

        effective_duration = max(duration_s, (total_frames / 10.0), 1.0)
        blink_rate_bpm = round((blink_count / effective_duration) * 60.0, 1)

        ear_stability = round(float(np.std(ears)), 4) if ears else 0.02

        # 2. Warmth & Facial Expressiveness
        smile_mean = round(float(np.mean(smile_ratios)), 3) if smile_ratios else 0.45
        # Smiling threshold AU12 > 0.52
        smiling_frames = sum(1 for s in smile_ratios if s > 0.52)
        smiling_ratio = round((smiling_frames / max(len(smile_ratios), 1)) * 100.0, 1)

        brow_mean = round(float(np.mean(brow_furrows)), 3) if brow_furrows else 0.20
        # Brow tension threshold AU4 inner brow distance < 0.175
        tense_frames = sum(1 for b in brow_furrows if b < 0.175)
        brow_tension_ratio = round((tense_frames / max(len(brow_furrows), 1)) * 100.0, 1)

        # Facial expressiveness score: standard deviation of smile & mouth movement
        smile_std = float(np.std(smile_ratios)) if smile_ratios else 0.0
        mar_std = float(np.std(mars)) if mars else 0.0
        # Dynamic variation in 0.015 - 0.06 is healthy animated speech
        dynamic_var = smile_std + mar_std
        expressiveness_score = min(100.0, max(30.0, 50.0 + (dynamic_var / 0.05) * 45.0))
        expressiveness_score = round(expressiveness_score, 1)

        # Warmth score: baseline 70, boosted by smiling, penalized by brow furrow
        warmth = 65.0 + (smiling_ratio * 0.40) - (brow_tension_ratio * 0.35)
        warmth_score = round(min(100.0, max(30.0, warmth)), 1)

        # 3. Head Movement & Poise (stability within frame)
        if len(nose_positions) >= 2:
            xs = [p[0] for p in nose_positions]
            ys = [p[1] for p in nose_positions]
            pos_std = math.sqrt(float(np.std(xs)) ** 2 + float(np.std(ys)) ** 2)
            # Low jitter (std < 0.03) = high poise (90-100); std > 0.10 = excessive fidgeting
            stability = 100.0 - (pos_std / 0.08) * 40.0
            head_stability_score = round(min(100.0, max(40.0, stability)), 1)
        else:
            head_stability_score = 80.0

        return FacialMetrics(
            frames_analyzed=total_frames,
            faces_detected_count=faces_detected,
            eye_contact_ratio=eye_contact_ratio,
            looking_away_ratio=looking_away_ratio,
            blink_count=blink_count,
            blink_rate_bpm=blink_rate_bpm,
            ear_stability=ear_stability,
            smile_ratio_mean=smile_mean,
            smiling_ratio=smiling_ratio,
            brow_furrow_mean=brow_mean,
            brow_tension_ratio=brow_tension_ratio,
            facial_expressiveness=expressiveness_score,
            head_stability_score=head_stability_score,
            warmth_score=warmth_score,
        )

    # -------------------------------------------------------------------------
    # 2. Vocal & Speech Analysis (DSP + Whisper)
    # -------------------------------------------------------------------------
    HALLUCINATION_PATTERNS = [
        re.compile(r"^\s*(mmm+|mm+|uh+|ah+|oh+|yeah|shh+|pfft|hm+|hmm+)\s*[\.\!\?]*\s*$", re.IGNORECASE),
        re.compile(r"(subtitles?\s+by|thank\s+you\s+for\s+watching|please\s+subscribe|amara\.org|translated\s+by|viewers\s+like\s+you)", re.IGNORECASE),
        re.compile(r"^\s*(you|bye|goodbye|okay|yes|no)\s*[\.\!\?]*\s*$", re.IGNORECASE),
    ]

    def detect_voice_activity(
        self, samples: Optional[np.ndarray], sample_rate: int = 16000
    ) -> Tuple[bool, float, float]:
        """Performs robust Voice Activity Detection (VAD) to distinguish speech from ambient silence/mic noise.

        Args:
            samples: Raw un-normalized 1D float32 audio waveform.
            sample_rate: Audio sampling frequency in Hz.

        Returns:
            Tuple of (has_speech: bool, active_speech_duration_s: float, peak_amplitude: float).
        """
        if samples is None or len(samples) < int(sample_rate * 0.25):
            return False, 0.0, 0.0

        # Evaluate on raw, un-normalized samples
        frame_len = int(sample_rate * 0.030)  # 30ms frames
        frames = [samples[i : i + frame_len] for i in range(0, len(samples) - frame_len + 1, frame_len)]
        if not frames:
            return False, 0.0, 0.0

        frame_energies = np.array([float(np.sqrt(np.mean(f**2))) for f in frames])
        peak_energy = float(np.max(frame_energies))
        peak_abs = float(np.max(np.abs(samples)))
        noise_floor = float(np.percentile(frame_energies, 20))

        # Conversational speech requires frames above ambient noise and above baseline threshold
        speech_thresh = max(0.015, noise_floor * 2.2)
        active_frames = np.sum(frame_energies >= speech_thresh)
        active_dur = (active_frames * frame_len) / float(sample_rate)

        # Genuine speech requires:
        # 1. Peak amplitude >= 0.025 (speech plosives/vowels significantly exceed mic floor)
        # 2. Peak frame RMS >= 0.015
        # 3. Minimum active speech duration >= 0.25 seconds
        has_speech = (peak_abs >= 0.025) and (peak_energy >= 0.015) and (active_dur >= 0.25)
        return has_speech, active_dur, peak_abs

    def analyze_vocal_and_speech(
        self, samples: Optional[np.ndarray], sample_rate: int = 16000
    ) -> Tuple[VocalMetrics, SpeechMetrics]:
        """Performs audio DSP feature extraction and local Whisper ASR transcription.

        Args:
            samples: 1D float32 audio waveform.
            sample_rate: Audio sampling frequency in Hz (typically 16000).

        Returns:
            Tuple of (VocalMetrics, SpeechMetrics).
        """
        if samples is None or len(samples) == 0:
            return VocalMetrics(), SpeechMetrics()

        duration_s = round(len(samples) / float(sample_rate), 2)

        # Pre-VAD Voice Activity Gate
        has_speech, active_dur, peak_abs = self.detect_voice_activity(samples, sample_rate)

        # If no speech was detected (silence or ambient room noise), return silent metrics
        if not has_speech:
            vocal_metrics = VocalMetrics(
                rms_energy_mean=round(float(np.sqrt(np.mean(samples**2))), 4),
                rms_energy_std=0.001,
                vocal_energy_score=25.0,  # Below conversational threshold
                pitch_mean_hz=0.0,
                pitch_std_hz=0.0,
                pitch_dynamic_score=30.0,
                is_monotone=True,
                words_per_minute=0.0,
                pacing_score=25.0,
                pause_ratio=1.0,  # 100% pause/silence
                speech_duration_s=duration_s,
            )
            speech_metrics = SpeechMetrics(
                transcript="",
                word_count=0,
                filler_words_count=0,
                filler_words_detected={},
                filler_word_ratio=0.0,
            )
            return vocal_metrics, speech_metrics

        # 1. DSP Prosody via reused feature_extraction.py
        prosody = extract_prosody_features(samples.tolist(), sample_rate=sample_rate)
        energy_mean = prosody.get("energy_mean", 0.0)
        energy_std = prosody.get("energy_std", 0.0)
        pitch_mean_hz = prosody.get("pitch_mean_hz", 120.0)
        pitch_std_hz = prosody.get("pitch_std_hz", 15.0)

        # Vocal energy score (0-100)
        # Healthy conversational speech normalized RMS is ~0.06 - 0.24
        if energy_mean < 0.025:
            vocal_energy_score = max(30.0, (energy_mean / 0.025) * 65.0)
        elif 0.05 <= energy_mean <= 0.28:
            vocal_energy_score = 90.0 + min(10.0, (energy_mean - 0.05) / 0.23 * 10.0)
        elif energy_mean > 0.40:
            vocal_energy_score = max(55.0, 95.0 - ((energy_mean - 0.40) / 0.40) * 35.0)
        else:
            vocal_energy_score = 78.0
        vocal_energy_score = round(vocal_energy_score, 1)

        # Pitch dynamics & monotone detection
        # Pitch standard deviation: < 14 Hz is flat/monotone; 22 - 50 Hz is dynamic/engaging
        is_monotone = pitch_std_hz < 14.0
        if 22.0 <= pitch_std_hz <= 52.0:
            pitch_dynamic_score = 92.0 + min(8.0, (pitch_std_hz - 22.0) / 30.0 * 8.0)
        elif pitch_std_hz < 14.0:
            pitch_dynamic_score = max(35.0, 45.0 + (pitch_std_hz / 14.0) * 25.0)
        elif 14.0 <= pitch_std_hz < 22.0:
            pitch_dynamic_score = 70.0 + ((pitch_std_hz - 14.0) / 8.0) * 20.0
        else:
            # Overly volatile (> 60 Hz)
            pitch_dynamic_score = max(70.0, 95.0 - ((pitch_std_hz - 52.0) / 30.0) * 20.0)
        pitch_dynamic_score = round(pitch_dynamic_score, 1)

        # Pause ratio (frame RMS silence detection with 25ms frames)
        frame_len = int(sample_rate * 0.025)
        silent_count = 0
        total_eval_frames = len(samples) // frame_len
        if total_eval_frames > 0:
            for i in range(total_eval_frames):
                frm = samples[i * frame_len : (i + 1) * frame_len]
                rms = compute_rms_energy(frm)
                if rms < 0.018:
                    silent_count += 1
            pause_ratio = round(silent_count / float(total_eval_frames), 3)
        else:
            pause_ratio = 0.0

        # 2. Whisper Speech-To-Text Transcription (with VAD protection)
        transcript = self._transcribe_audio(samples, sample_rate)
        words = [w for w in re.split(r"\s+", transcript) if w.strip()]
        word_count = len(words)

        # Pacing (Words Per Minute / WPM)
        # Optimal interview target range: 130 - 160 WPM
        words_per_minute = round((word_count / max(duration_s, 0.5)) * 60.0, 1)

        if word_count == 0:
            pacing_score = 25.0
        elif 130.0 <= words_per_minute <= 160.0:
            pacing_score = 96.0 + min(4.0, (1.0 - abs(words_per_minute - 145.0) / 15.0) * 4.0)
        elif 115.0 <= words_per_minute < 130.0:
            pacing_score = 82.0 + ((words_per_minute - 115.0) / 15.0) * 12.0
        elif 160.0 < words_per_minute <= 175.0:
            pacing_score = 82.0 + ((175.0 - words_per_minute) / 15.0) * 12.0
        elif 95.0 <= words_per_minute < 115.0:
            pacing_score = 62.0 + ((words_per_minute - 95.0) / 20.0) * 18.0
        elif 175.0 < words_per_minute <= 195.0:
            pacing_score = 62.0 + ((195.0 - words_per_minute) / 20.0) * 18.0
        else:
            pacing_score = max(35.0, 55.0 - (abs(words_per_minute - 145.0) / 100.0) * 20.0)
        pacing_score = round(pacing_score, 1)

        # 3. Filler Word Detection
        filler_matches = FILLER_REGEX.findall(transcript.lower())
        filler_counts: Dict[str, int] = {}
        for f in filler_matches:
            cleaned_f = re.sub(r"\s+", " ", f.strip().lower())
            filler_counts[cleaned_f] = filler_counts.get(cleaned_f, 0) + 1

        filler_words_count = sum(filler_counts.values())
        filler_word_ratio = round((filler_words_count / max(word_count, 1)) * 100.0, 1)

        vocal_metrics = VocalMetrics(
            rms_energy_mean=round(energy_mean, 4),
            rms_energy_std=round(energy_std, 4),
            vocal_energy_score=vocal_energy_score,
            pitch_mean_hz=round(pitch_mean_hz, 1),
            pitch_std_hz=round(pitch_std_hz, 1),
            pitch_dynamic_score=pitch_dynamic_score,
            is_monotone=is_monotone,
            words_per_minute=words_per_minute,
            pacing_score=pacing_score,
            pause_ratio=pause_ratio,
            speech_duration_s=duration_s,
        )

        speech_metrics = SpeechMetrics(
            transcript=transcript,
            word_count=word_count,
            filler_words_count=filler_words_count,
            filler_words_detected=filler_counts,
            filler_word_ratio=filler_word_ratio,
        )

        return vocal_metrics, speech_metrics

    def _transcribe_audio(self, samples: np.ndarray, sample_rate: int = 16000) -> str:
        """Internal Whisper helper with VAD gating and hallucination suppression."""
        if samples is None or len(samples) == 0:
            return ""

        # Step 1: Pre-VAD Voice Activity Gate
        has_speech, active_dur, peak_abs = self.detect_voice_activity(samples, sample_rate)
        if not has_speech:
            return ""

        asr = get_asr_pipeline()
        if asr is None:
            return ""

        try:
            # Standardize to 1D float32
            if samples.ndim > 1:
                samples = np.mean(samples, axis=1)
            samples = samples.astype(np.float32)

            # Remove DC bias
            samples = samples - float(np.mean(samples))

            # Only normalize if genuine speech energy exists
            if peak_abs > 0.025:
                samples = samples / peak_abs * 0.90

            # Trim silence around speech
            try:
                trimmed = trim_silence(samples, threshold=0.025)
                if len(trimmed) > sample_rate * 0.25:
                    samples = trimmed
            except Exception:
                pass

            # Fast transcription
            result = asr(samples, generate_kwargs={"max_new_tokens": 128})
            raw_text = result.get("text", "").strip()
            clean_text = re.sub(r"\s+", " ", raw_text).strip()

            # Ignore pure punctuation hallucinations
            if re.match(r"^[\s\.\,\!\?\-]+$", clean_text):
                return ""

            # Filter known Whisper silence hallucinations
            for pat in self.HALLUCINATION_PATTERNS:
                if pat.search(clean_text) and (active_dur < 0.6 or len(clean_text.split()) <= 2):
                    return ""

            return clean_text
        except Exception as e:
            print(f"[InterviewAnalyzer] Transcription notice: {e}")
            return ""

    # -------------------------------------------------------------------------
    # 3. Coaching Synthesis & Scoring
    # -------------------------------------------------------------------------
    def synthesize_coaching(
        self,
        facial: FacialMetrics,
        vocal: VocalMetrics,
        speech: SpeechMetrics,
    ) -> AnalysisResult:
        """Synthesizes raw metrics into 0-100 rubric scores and actionable interview advice.

        Args:
            facial: Facial metrics.
            vocal: Vocal prosody metrics.
            speech: Transcript and filler word metrics.

        Returns:
            AnalysisResult with confidence, clarity, engagement, overall scores,
            strengths, and actionable improvements.
        """
        # 1. Confidence Score (0-100)
        # Projection (25%) + Pitch dynamic (20%) + Eye contact & poise (35%) + Low hesitation (20%)
        filler_penalty = min(35.0, speech.filler_word_ratio * 4.0)
        hesitation_score = max(40.0, 100.0 - filler_penalty)

        eye_contact_component = facial.eye_contact_ratio if facial.faces_detected_count > 0 else 75.0
        poise_component = (
            (eye_contact_component * 0.65 + facial.head_stability_score * 0.35)
            if facial.faces_detected_count > 0
            else 75.0
        )

        confidence = (
            0.25 * vocal.vocal_energy_score
            + 0.20 * vocal.pitch_dynamic_score
            + 0.35 * poise_component
            + 0.20 * hesitation_score
        )
        confidence_score = round(min(100.0, max(25.0, confidence)), 1)

        # 2. Clarity Score (0-100)
        # Pacing alignment (40%) + Articulation/No fillers (35%) + Structured pauses (25%)
        # Pause balance: ideal pause ratio is 0.15 - 0.30
        if 0.12 <= vocal.pause_ratio <= 0.32:
            pause_score = 95.0
        elif vocal.pause_ratio > 0.42:
            pause_score = max(45.0, 95.0 - ((vocal.pause_ratio - 0.42) / 0.40) * 50.0)
        elif vocal.pause_ratio < 0.08:
            pause_score = max(55.0, 95.0 - ((0.08 - vocal.pause_ratio) / 0.08) * 35.0)
        else:
            pause_score = 80.0

        clarity = (
            0.40 * vocal.pacing_score
            + 0.35 * max(35.0, 100.0 - (speech.filler_word_ratio * 5.0))
            + 0.25 * pause_score
        )
        clarity_score = round(min(100.0, max(25.0, clarity)), 1)

        # 3. Engagement Score (0-100)
        # Facial Warmth (35%) + Pitch Inflection (35%) + Facial Animation (15%) + Eye Contact (15%)
        if facial.faces_detected_count > 0:
            facial_warmth_comp = facial.warmth_score
            facial_anim_comp = facial.facial_expressiveness
            eye_contact_comp = facial.eye_contact_ratio
        else:
            facial_warmth_comp = 72.0
            facial_anim_comp = 70.0
            eye_contact_comp = 72.0

        engagement = (
            0.35 * facial_warmth_comp
            + 0.35 * vocal.pitch_dynamic_score
            + 0.15 * facial_anim_comp
            + 0.15 * eye_contact_comp
        )
        engagement_score = round(min(100.0, max(25.0, engagement)), 1)

        # 4. Overall Interview Readiness Score (0-100)
        overall = 0.35 * confidence_score + 0.35 * clarity_score + 0.30 * engagement_score
        overall_delivery_score = round(min(100.0, max(25.0, overall)), 1)

        # 5. Rule-Based Dynamic Coaching Feedback
        strengths, improvements = self._generate_actionable_feedback(
            facial, vocal, speech, confidence_score, clarity_score, engagement_score
        )

        return AnalysisResult(
            overall_delivery_score=overall_delivery_score,
            confidence_score=confidence_score,
            clarity_score=clarity_score,
            engagement_score=engagement_score,
            scores={
                "overall_delivery_score": overall_delivery_score,
                "confidence_score": confidence_score,
                "clarity_score": clarity_score,
                "engagement_score": engagement_score,
            },
            metrics={
                "facial": asdict(facial),
                "vocal": asdict(vocal),
                "speech": asdict(speech),
            },
            strengths=strengths,
            actionable_improvements=improvements,
        )

    def _generate_actionable_feedback(
        self,
        facial: FacialMetrics,
        vocal: VocalMetrics,
        speech: SpeechMetrics,
        conf_score: float,
        clarity_score: float,
        engage_score: float,
    ) -> Tuple[List[str], List[str]]:
        """Generates specific, personalized strengths and improvement advice based on rubrics."""
        strengths: List[str] = []
        improvements: List[str] = []

        # Silent / No words spoken feedback
        if speech.word_count == 0:
            improvements.append(
                "No spoken words detected: audio captured only ambient background silence. Speak clearly at normal conversational volume to receive speech, pacing, and filler word feedback."
            )

        # Pacing feedback
        elif 130.0 <= vocal.words_per_minute <= 160.0:
            strengths.append(
                f"Optimal conversational pacing ({vocal.words_per_minute:.0f} WPM) within the ideal 130–160 WPM interview sweet spot."
            )
        elif vocal.words_per_minute > 165.0:
            improvements.append(
                f"Moderate speech tempo: currently at {vocal.words_per_minute:.0f} WPM (ideal: 130–160 WPM). Slow down slightly to allow your answers to resonate."
            )
        elif 0 < vocal.words_per_minute < 115.0:
            improvements.append(
                f"Pick up conversational pace: speaking at {vocal.words_per_minute:.0f} WPM can sound hesitant. Aim for a brisk, energetic delivery."
            )

        # Eye contact & Poise feedback
        if facial.faces_detected_count > 0:
            if facial.eye_contact_ratio >= 75.0:
                strengths.append(
                    f"Strong, steady eye contact ({facial.eye_contact_ratio:.0f}%) conveying executive presence and authenticity."
                )
            elif facial.looking_away_ratio >= 35.0:
                improvements.append(
                    f"Increase direct eye contact: looking away {facial.looking_away_ratio:.0f}% of the time. Position your webcam at eye level to maintain natural rapport."
                )

            if facial.head_stability_score >= 85.0:
                strengths.append("Controlled head posture and poise with minimal distracting movement.")
            elif facial.head_stability_score < 70.0:
                improvements.append("Reduce frequent head swaying or fidgeting to project calm confidence.")

            # Warmth & Facial expressiveness
            if facial.smiling_ratio >= 25.0:
                strengths.append("Warm and approachable facial affect, fostering an engaging interview connection.")
            elif facial.brow_tension_ratio > 30.0:
                improvements.append("Noticeable brow furrowing detected. Relax your forehead and jaw to avoid appearing overly tense or stressed.")

        # Pitch dynamics & Monotone feedback
        if not vocal.is_monotone and vocal.pitch_std_hz >= 22.0:
            strengths.append(
                f"Engaging vocal inflection ({vocal.pitch_std_hz:.1f} Hz variance) effectively preventing a monotone presentation."
            )
        elif vocal.is_monotone:
            improvements.append(
                f"Vocal delivery is relatively monotone ({vocal.pitch_std_hz:.1f} Hz variance). Practice inflecting on key technical metrics and achievements."
            )

        # Vocal Energy & Projection feedback
        if vocal.vocal_energy_score >= 88.0:
            strengths.append("Robust vocal projection and steady energy throughout responses.")
        elif vocal.vocal_energy_score < 68.0:
            improvements.append("Increase vocal volume and projection; soft speech can diminish perceived authority.")

        # Filler Words feedback
        if speech.filler_words_count == 0 and speech.word_count > 10:
            strengths.append("Exceptional verbal discipline with zero detected filler words.")
        elif speech.filler_words_count > 0:
            filler_list = ", ".join(f"'{k}' ({v}x)" for k, v in list(speech.filler_words_detected.items())[:3])
            improvements.append(
                f"Minimize filler words: detected {speech.filler_words_count} filler(s) including {filler_list}. Replace fillers with deliberate 1-second silent pauses."
            )

        # Fallbacks to ensure at least 2 strengths and 2 improvements
        if len(strengths) < 2:
            if conf_score >= 70.0:
                strengths.append("Well-structured, decisive responses demonstrating solid overall preparation.")
            if engage_score >= 70.0:
                strengths.append("Attentive communicative posture that keeps the interviewer engaged.")
            if len(strengths) < 2:
                strengths.append("Clear vocal clarity and balanced acoustic delivery.")

        if len(improvements) < 2:
            if vocal.pause_ratio > 0.38:
                improvements.append("Excessive silent pauses detected between phrases. Group your thoughts in advance using the STAR method.")
            elif vocal.pause_ratio < 0.10 and speech.word_count > 15:
                improvements.append("Incorporate brief, natural pauses after key points to let critical takeaways sink in.")
            else:
                improvements.append("Frame your answers with the STAR format (Situation, Task, Action, Result) for maximum structural punch.")

        return strengths[:3], improvements[:3]

    # -------------------------------------------------------------------------
    # High-Level End-to-End Orchestrator
    # -------------------------------------------------------------------------
    def analyze(
        self,
        video_path: Optional[str] = None,
        audio_bytes: Optional[bytes] = None,
        frames: Optional[List[np.ndarray]] = None,
    ) -> AnalysisResult:
        """Executes full end-to-end multimodal facial & vocal analysis.

        Args:
            video_path: Local filesystem path to a video file.
            audio_bytes: In-memory byte buffer of audio (WAV, MP3, WebM).
            frames: Pre-extracted or webcam image frames (BGR numpy arrays).

        Returns:
            AnalysisResult with normalized scores, metrics, and actionable feedback.
        """
        video_duration = 0.0

        # Step 1: If video_path provided, demux audio & sample video frames
        if video_path and os.path.exists(video_path):
            extracted_samples, sr, audio_dur = self.extract_audio_from_video(video_path)
            extracted_frames, vid_dur = self.sample_video_frames(video_path, max_samples=30)
            samples = extracted_samples
            sample_rate = sr
            video_duration = max(audio_dur, vid_dur)
            frames = extracted_frames
        else:
            samples = None
            sample_rate = 16000

        # Step 2: If audio_bytes provided, parse into float32 samples
        if audio_bytes and (samples is None or len(samples) == 0):
            parsed_samples, sr, audio_dur = self.parse_wav_bytes(audio_bytes)
            samples = parsed_samples
            sample_rate = sr
            if video_duration <= 0.0:
                video_duration = audio_dur

        # Step 3: Run Facial Analysis
        facial_metrics = self.analyze_facial_frames(frames or [], duration_s=video_duration)

        # Step 4: Run Vocal & Speech Analysis
        vocal_metrics, speech_metrics = self.analyze_vocal_and_speech(samples, sample_rate)

        # Step 5: Synthesize Coaching Scores & Actionable Feedback
        return self.synthesize_coaching(facial_metrics, vocal_metrics, speech_metrics)
