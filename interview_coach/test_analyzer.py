"""Standalone CLI Test Runner for Interview Preparation AI Coach Engine.

Runs comprehensive verification on:
1. Filler word scanner & regex accuracy
2. Vocal DSP prosody extraction & scoring
3. Facial Action Unit analysis & poise geometry
4. End-to-end multimodal pipeline on video clip
5. Flask REST API endpoint verification (POST /analyze-interview)

Usage:
    python interview_coach/test_analyzer.py
"""

import base64
import json
import os
import subprocess
import sys
import time
import wave
import numpy as np

# Ensure parent and module directories are in sys.path
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_CURRENT_DIR)
for _p in [_CURRENT_DIR, _PARENT_DIR, r"C:\tcs", r"C:\multimodel-emotionAI"]:
    if os.path.exists(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from interview_analyzer import (
    FILLER_REGEX,
    FacialMetrics,
    InterviewAnalyzer,
    SpeechMetrics,
    VocalMetrics,
)
from server import app


def generate_synthetic_audio(duration_s: float = 3.0, sample_rate: int = 16000) -> bytes:
    """Generates synthetic multi-tone speech-like audio with pause structure."""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    # Fundamental pitch ~160 Hz with harmonics at 320 Hz and 480 Hz
    wave_signal = (
        0.5 * np.sin(2 * np.pi * 160 * t)
        + 0.25 * np.sin(2 * np.pi * 320 * t)
        + 0.15 * np.sin(2 * np.pi * 480 * t)
    )
    # Add natural amplitude envelope and a 0.5s pause in the middle
    envelope = 0.5 * (1 + np.sin(2 * np.pi * 1.5 * t))
    wave_signal = wave_signal * envelope
    # Insert deliberate pause
    pause_start = int(sample_rate * 1.2)
    pause_end = int(sample_rate * 1.7)
    wave_signal[pause_start:pause_end] = 0.0

    # Convert to 16-bit PCM WAV bytes
    pcm_data = (wave_signal * 32767).astype(np.int16).tobytes()
    import io

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return buf.getvalue()


def test_filler_word_scanner():
    """Validates regex filler word detection on sample interview transcripts."""
    print("\n[1/5] Testing Filler Word Scanner...")
    test_text = (
        "Well, um, I actually believe that, like, our team sort of succeeded, you know? "
        "Basically, uh, we exceeded our KPI targets."
    )
    matches = FILLER_REGEX.findall(test_text.lower())
    detected = [m.strip().lower() for m in matches]
    print(f"  Detected Fillers: {detected}")

    assert "um" in detected, "Failed to detect 'um'"
    assert "actually" in detected, "Failed to detect 'actually'"
    assert "like" in detected, "Failed to detect 'like'"
    assert "sort of" in detected, "Failed to detect 'sort of'"
    assert "you know" in detected, "Failed to detect 'you know'"
    assert "basically" in detected, "Failed to detect 'basically'"
    assert "uh" in detected, "Failed to detect 'uh'"
    print("  -> PASSED: All filler words identified accurately.")


def test_vocal_and_prosody_analysis(analyzer: InterviewAnalyzer):
    """Validates audio DSP prosody extraction and score bounds."""
    print("\n[2/5] Testing Vocal & Audio Prosody Analysis...")
    wav_bytes = generate_synthetic_audio(duration_s=3.0)
    samples, sr, dur = analyzer.parse_wav_bytes(wav_bytes)

    assert samples is not None and len(samples) > 0, "Failed to parse WAV samples"
    assert dur > 2.5, f"Expected ~3.0s duration, got {dur:.2f}s"

    vocal, speech = analyzer.analyze_vocal_and_speech(samples, sr)

    print(f"  Vocal RMS Energy Mean : {vocal.rms_energy_mean:.4f}")
    print(f"  Vocal Energy Score    : {vocal.vocal_energy_score:.1f}/100")
    print(f"  Pitch Mean            : {vocal.pitch_mean_hz:.1f} Hz")
    print(f"  Pitch Dynamic Score   : {vocal.pitch_dynamic_score:.1f}/100")
    print(f"  Pause Ratio           : {vocal.pause_ratio:.3f}")
    print(f"  Pacing Score          : {vocal.pacing_score:.1f}/100")

    assert 0.0 <= vocal.vocal_energy_score <= 100.0, "Vocal energy score out of bounds"
    assert 0.0 <= vocal.pitch_dynamic_score <= 100.0, "Pitch dynamic score out of bounds"
    assert 0.0 <= vocal.pacing_score <= 100.0, "Pacing score out of bounds"
    assert 0.0 <= vocal.pause_ratio <= 1.0, "Pause ratio out of bounds"
    print("  -> PASSED: Audio prosody and scoring logic verified.")


def test_silence_rejection(analyzer: InterviewAnalyzer):
    """Validates that ambient microphone noise or zero-speech clips are NOT hallucinated as words."""
    print("\n[VAD Test] Testing Silence & Ambient Noise Rejection...")
    # Generate 3 seconds of soft ambient microphone noise
    noise_signal = np.random.normal(0, 0.005, 16000 * 3).astype(np.float32)
    has_speech, dur, peak = analyzer.detect_voice_activity(noise_signal, 16000)
    print(f"  Noise detected as speech: {has_speech} (peak: {peak:.4f}, active duration: {dur:.2f}s)")
    assert not has_speech, "Ambient noise should NOT trigger voice activity detection!"

    vocal, speech = analyzer.analyze_vocal_and_speech(noise_signal, 16000)
    print(f"  Transcribed text on noise: {repr(speech.transcript)}")
    print(f"  Word count on noise       : {speech.word_count}")
    assert speech.transcript == "", f"Expected empty transcript on silence, got: '{speech.transcript}'"
    assert speech.word_count == 0, f"Expected 0 words on silence, got: {speech.word_count}"
    assert vocal.pause_ratio == 1.0, f"Expected 100% pause ratio on silence, got: {vocal.pause_ratio}"
    print("  -> PASSED: Ambient noise rejected cleanly; zero hallucinated words.")


def test_facial_analysis(analyzer: InterviewAnalyzer):
    """Validates MediaPipe FaceMesh AU calculation and poise metrics."""
    print("\n[3/5] Testing Facial Action Unit & Poise Analysis...")
    # Attempt to load benchmark thumbnail or generate synthetic frame
    img_path = r"C:\multimodel-emotionAI\backend\benchmark_thumbnail.jpg"
    frames = []
    if os.path.exists(img_path):
        import cv2
        img = cv2.imread(img_path)
        if img is not None:
            frames = [img] * 10

    facial = analyzer.analyze_facial_frames(frames, duration_s=3.0)

    print(f"  Frames Analyzed       : {facial.frames_analyzed}")
    print(f"  Faces Detected Count  : {facial.faces_detected_count}")
    print(f"  Eye Contact Ratio     : {facial.eye_contact_ratio:.1f}%")
    print(f"  Smile Ratio Mean      : {facial.smile_ratio_mean:.3f}")
    print(f"  Warmth Score          : {facial.warmth_score:.1f}/100")
    print(f"  Head Stability Score  : {facial.head_stability_score:.1f}/100")

    assert 0.0 <= facial.eye_contact_ratio <= 100.0, "Eye contact ratio out of bounds"
    assert 0.0 <= facial.warmth_score <= 100.0, "Warmth score out of bounds"
    assert 0.0 <= facial.head_stability_score <= 100.0, "Head stability score out of bounds"
    print("  -> PASSED: Facial Action Units and gaze metrics verified.")


def test_end_to_end_multimodal_pipeline(analyzer: InterviewAnalyzer) -> dict:
    """Executes full end-to-end multimodal interview coaching pipeline on a video file."""
    print("\n[4/5] Testing Full End-to-End Multimodal Pipeline on Video Clip...")
    sample_mp4 = os.path.join(_CURRENT_DIR, "sample_interview.mp4")

    # If sample clip does not exist, synthesize one
    if not os.path.exists(sample_mp4):
        print("  Synthesizing sample interview MP4...")
        wav_bytes = generate_synthetic_audio(duration_s=2.5)
        temp_wav = os.path.join(_CURRENT_DIR, "temp_synth.wav")
        with open(temp_wav, "wb") as f:
            f.write(wav_bytes)

        img_path = r"C:\multimodel-emotionAI\backend\benchmark_thumbnail.jpg"
        ffmpeg_exe = analyzer.ffmpeg_exe
        cmd = [
            ffmpeg_exe,
            "-y",
            "-loop",
            "1",
            "-i",
            img_path,
            "-i",
            temp_wav,
            "-c:v",
            "libx264",
            "-t",
            "2.5",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            sample_mp4,
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
            except Exception:
                pass

    t0 = time.time()
    result = analyzer.analyze(video_path=sample_mp4)
    elapsed = time.time() - t0
    res_dict = result.to_dict()

    print(f"  Execution Time        : {elapsed:.2f} seconds")
    print(f"  Overall Delivery Score: {res_dict['overall_delivery_score']}/100")
    print(f"  Confidence Score      : {res_dict['confidence_score']}/100")
    print(f"  Clarity Score         : {res_dict['clarity_score']}/100")
    print(f"  Engagement Score      : {res_dict['engagement_score']}/100")
    print(f"  Top Strengths ({len(res_dict['strengths'])}):")
    for s in res_dict["strengths"]:
        print(f"    + {s}")
    print(f"  Actionable Improvements ({len(res_dict['actionable_improvements'])}):")
    for imp in res_dict["actionable_improvements"]:
        print(f"    - {imp}")

    assert 0.0 <= res_dict["overall_delivery_score"] <= 100.0, "Overall score out of bounds"
    assert 0.0 <= res_dict["confidence_score"] <= 100.0, "Confidence score out of bounds"
    assert 0.0 <= res_dict["clarity_score"] <= 100.0, "Clarity score out of bounds"
    assert 0.0 <= res_dict["engagement_score"] <= 100.0, "Engagement score out of bounds"
    assert len(res_dict["strengths"]) >= 2, "Expected at least 2 strengths"
    assert len(res_dict["actionable_improvements"]) >= 2, "Expected at least 2 improvements"
    assert "facial" in res_dict["metrics"], "Missing facial metrics"
    assert "vocal" in res_dict["metrics"], "Missing vocal metrics"
    assert "speech" in res_dict["metrics"], "Missing speech metrics"

    print("  -> PASSED: Multimodal analysis completed with valid scores and coaching advice.")
    return res_dict


def test_flask_api_endpoints():
    """Validates the Flask REST API endpoint using the Flask test client."""
    print("\n[5/5] Testing Flask API Endpoint (POST /analyze-interview)...")
    client = app.test_client()

    # Health check
    res_health = client.get("/health")
    assert res_health.status_code == 200, f"Expected 200 from /health, got {res_health.status_code}"
    print("  -> GET /health returned 200 OK.")

    # Base64 audio payload test
    wav_bytes = generate_synthetic_audio(duration_s=2.0)
    b64_audio = base64.b64encode(wav_bytes).decode("utf-8")

    t0 = time.time()
    res_api = client.post(
        "/analyze-interview",
        json={"audio_base64": b64_audio},
        content_type="application/json",
    )
    elapsed = time.time() - t0

    assert res_api.status_code == 200, f"Expected 200, got {res_api.status_code}: {res_api.data}"
    json_data = res_api.get_json()
    assert json_data.get("success") is True, "API response success flag was not True"
    assert "overall_delivery_score" in json_data, "Missing overall_delivery_score in response"
    assert "metrics" in json_data, "Missing metrics in response"
    assert "strengths" in json_data, "Missing strengths in response"
    assert "actionable_improvements" in json_data, "Missing actionable_improvements in response"

    print(f"  -> POST /analyze-interview returned 200 OK in {elapsed:.2f}s.")
    print("  -> PASSED: REST API test completed successfully.")


def main():
    """Main execution function running all tests sequentially."""
    print("=" * 70)
    print(" INTERVIEW PREPARATION AI COACH - FACIAL & VOCAL ENGINE TEST RUNNER")
    print("=" * 70)

    start_total = time.time()
    analyzer = InterviewAnalyzer()

    # Run tests
    test_filler_word_scanner()
    test_vocal_and_prosody_analysis(analyzer)
    test_silence_rejection(analyzer)
    test_facial_analysis(analyzer)
    final_output = test_end_to_end_multimodal_pipeline(analyzer)
    test_flask_api_endpoints()

    total_time = time.time() - start_total

    print("\n" + "=" * 70)
    print(" FORMATTED ENGINE JSON OUTPUT")
    print("=" * 70)
    print(json.dumps(final_output, indent=2))

    print("\n" + "=" * 70)
    print(f" ALL TESTS PASSED! Total Execution Time: {total_time:.2f} seconds")
    print("=" * 70)


if __name__ == "__main__":
    main()
