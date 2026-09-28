"""Interview Preparation AI Coach - Lightweight Flask API Server.

Exposes REST endpoint `POST /analyze-interview` for ingesting candidate video files
or base64-encoded audio/video payloads, executing multimodal facial & vocal analysis,
and returning coaching scores, prosody metrics, transcript, and actionable feedback.
"""

import base64
import os
import sys
import tempfile
import traceback
from typing import Any, Dict

from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS

# Ensure parent and module directories are in path
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_CURRENT_DIR)
for _p in [_CURRENT_DIR, _PARENT_DIR, r"C:\tcs", r"C:\multimodel-emotionAI"]:
    if os.path.exists(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from interview_analyzer import InterviewAnalyzer

app = Flask(__name__, template_folder=os.path.join(_CURRENT_DIR, "templates"))
app.config["MAX_CONTENT_LENGTH"] = 150 * 1024 * 1024  # 150MB maximum payload
CORS(app)

# Singleton analyzer instance
_ANALYZER = None


def get_analyzer() -> InterviewAnalyzer:
    """Returns singleton instance of InterviewAnalyzer."""
    global _ANALYZER
    if _ANALYZER is None:
        _ANALYZER = InterviewAnalyzer()
    return _ANALYZER


@app.route("/", methods=["GET"])
def index():
    """Serves the interactive Interview AI Coach testing dashboard."""
    return render_template("index.html")


@app.route("/sample-video", methods=["GET"])
def sample_video():
    """Serves the demo interview MP4 video clip."""
    return send_from_directory(_CURRENT_DIR, "sample_interview.mp4", mimetype="video/mp4")


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint confirming API operational status."""
    return jsonify(
        {
            "status": "healthy",
            "service": "Interview Preparation AI Coach Backend",
            "version": "1.0.0",
            "endpoints": ["GET /", "POST /analyze-interview", "GET /sample-video", "GET /health"],
        }
    )


@app.route("/analyze-interview", methods=["POST"])
def analyze_interview():
    """Analyzes an uploaded video file or base64 audio/video payload.

    Accepts:
        - Multipart Form-Data with 'file' or 'video' or 'audio'
        - JSON payload with 'video_base64', 'audio_base64', or 'video_path'

    Returns:
        JSON response with overall_delivery_score, confidence_score, clarity_score,
        engagement_score, raw metrics breakdown, transcript, and coaching feedback.
    """
    analyzer = get_analyzer()
    temp_files_to_cleanup = []

    try:
        video_path = None
        audio_bytes = None

        # Case 1: Multipart file upload
        if request.files:
            file_obj = (
                request.files.get("file")
                or request.files.get("video")
                or request.files.get("audio")
            )
            if file_obj and file_obj.filename:
                orig_filename = file_obj.filename.lower()
                suffix = os.path.splitext(orig_filename)[1] or ".mp4"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
                    file_obj.save(tf.name)
                    video_path = tf.name
                    temp_files_to_cleanup.append(video_path)

        # Case 2: JSON payload
        elif request.is_json:
            data: Dict[str, Any] = request.get_json(silent=True) or {}

            # Direct file path
            if "video_path" in data and os.path.exists(data["video_path"]):
                video_path = data["video_path"]

            # Base64 video
            elif "video_base64" in data or "video" in data:
                b64_str = data.get("video_base64") or data.get("video")
                if "," in b64_str:
                    b64_str = b64_str.split(",", 1)[1]
                video_bytes = base64.b64decode(b64_str)
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
                    tf.write(video_bytes)
                    video_path = tf.name
                    temp_files_to_cleanup.append(video_path)

            # Base64 audio
            elif "audio_base64" in data or "audio" in data:
                b64_str = data.get("audio_base64") or data.get("audio")
                if "," in b64_str:
                    b64_str = b64_str.split(",", 1)[1]
                audio_bytes = base64.b64decode(b64_str)

        # Case 3: Raw binary stream
        elif request.data:
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tf:
                tf.write(request.data)
                video_path = tf.name
                temp_files_to_cleanup.append(video_path)

        if not video_path and not audio_bytes:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "No media provided. Submit video/audio via multipart file or JSON base64 payload.",
                    }
                ),
                400,
            )

        # Run analysis
        result = analyzer.analyze(video_path=video_path, audio_bytes=audio_bytes)
        return jsonify(result.to_dict()), 200

    except Exception as e:
        traceback.print_exc()
        return (
            jsonify(
                {
                    "success": False,
                    "error": str(e),
                }
            ),
            500,
        )

    finally:
        # Secure temporary file cleanup
        for tmp_file in temp_files_to_cleanup:
            if os.path.exists(tmp_file):
                try:
                    os.remove(tmp_file)
                except Exception:
                    pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print(f"Starting Interview Preparation AI Coach Server on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
