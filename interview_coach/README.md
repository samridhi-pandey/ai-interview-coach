# Interview Preparation AI Coach

A self-contained, high-performance **Facial & Vocal Analysis Engine** for real-time interview preparation and practice.

## Core Architecture & Capabilities

### 1. Facial Analysis (MediaPipe FaceMesh)
- **Eye Contact & Poise**: 3D geometric tracking of gaze yaw/pitch relative to the camera lens.
- **Blink Rate Monitoring**: Identifies blink dynamics via Eye Aspect Ratio (EAR < 0.21) and computes Blinks Per Minute (BPM).
- **Warmth & Affect**: Calculates Lip Corner Puller (AU12) smile ratio and Inner Brow Corrugator (AU4) furrowing/tension.
- **Head Poise Stability**: Evaluates postural sway and fidgeting variance across video frames.

### 2. Vocal Prosody & Speech Articulation (Audio DSP + Whisper ASR)
- **Vocal Energy & Projection**: RMS energy computation across conversational dynamic range.
- **Pitch Dynamic & Monotone Detection**: Autocorrelation fundamental frequency (F0) tracking. Flags flat, robotic delivery versus lively vocal inflection.
- **Conversational Pacing (WPM)**: Calculates Words Per Minute against the optimal 130–160 WPM interview window.
- **Speech Hesitation & Filler Word Scanner**: Regex detector identifying filler words (`um`, `uh`, `like`, `you know`, `actually`, `basically`, `sort of`, `kind of`, `literally`).
- **Voice Activity Detection (VAD) & Anti-Hallucination Filter**: Gated energy analysis that prevents Whisper from hallucinating words on ambient room silence.

### 3. Coaching Synthesis
- **Normalized 0–100 Scores**: `confidence_score`, `clarity_score`, `engagement_score`, and `overall_delivery_score`.
- **Actionable Strategic Feedback**: Automatically synthesizes top candidate strengths and priority areas for improvement.

---

## Directory Structure

```text
interview_coach/
├── interview_analyzer.py   # Core multimodal analysis pipeline
├── server.py               # Flask REST API + UI dashboard server (:5001)
├── test_analyzer.py        # Standalone test runner (VAD, prosody, facial, API)
├── sample_interview.mp4    # Verified demo clip for instant testing
├── requirements.txt        # Minimal dependencies
├── templates/
│   └── index.html          # Interactive dark-mode dashboard (Tailwind CDN)
└── utils/
    ├── __init__.py
    └── feature_extraction.py # Facial geometry Action Units & DSP signal processing
```

---

## Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Standalone Tests
```bash
python test_analyzer.py
```

### 3. Start the Web Dashboard & API Server
```bash
python server.py
```
Open **http://localhost:5001** in your browser to:
- Practice with live webcam recording.
- Upload any video/audio interview clip.
- Run one-click demo tests and view real-time scorecards.

---

## API Endpoints

- `GET /` - Interactive testing dashboard.
- `GET /health` - Service health status.
- `GET /sample-video` - Serves the demo interview clip.
- `POST /analyze-interview` - Accepts multipart video file (`file`) or base64 JSON payload (`video_base64`, `audio_base64`). Returns complete rubric scores, prosody/facial metrics, transcript, and coaching feedback.
