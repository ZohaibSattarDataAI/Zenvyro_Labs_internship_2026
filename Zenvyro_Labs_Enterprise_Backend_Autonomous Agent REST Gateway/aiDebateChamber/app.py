"""
app.py
------
Flask API gateway for the Autonomous AI Debate Chamber.

Routes:
    POST /api/debate/start                -> resets state, returns Agent A's opening statement
    POST /api/debate/next-turn            -> generates the next agent's turn (with memory)
    POST /api/machine-learning/train      -> trains the RandomForest judge on mock data
    POST /api/machine-learning/evaluate   -> scores both agents' full arguments and picks a winner

All heavy lifting lives in services/aiService.py (Ollama calls) and
services/mlJudge.py (feature extraction + regression). This file is just
the routing/orchestration layer.
"""

import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from services.aiService import DebateConductor
from services.mlJudge import DebateRegressionJudge, generate_mock_dataset
from services.aiService import DebateConductor
import services.aiService as ai

print("Loaded aiService from:")
print(ai.__file__)

# The frontend (index.html, css/, js/) lives in ./frontend relative to this file.
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app)  # Also allow the frontend to be opened as a standalone file (file://) and still call this API.


@app.route("/")
def serve_index():
    """Serves frontend/index.html so http://127.0.0.1:5000/ works directly."""
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:filename>")
def serve_static_asset(filename):
    """
    Serves everything else under frontend/ (css/, js/, images, demo.html, etc.)
    so relative paths like 'js/app.js' and 'css/demo.css' resolve correctly
    when the page is loaded from this server instead of opened as a file.
    """
    return send_from_directory(FRONTEND_DIR, filename)

# System modules — single shared instances for the life of the server.
conductor = DebateConductor()
ml_judge = DebateRegressionJudge()

# Accumulates full-argument text per agent for the current debate, so the
# ML judge can evaluate the whole performance at the end, not just one turn.
full_transcript = {"A": [], "B": []}


@app.route("/api/debate/start", methods=["POST"])
def start_debate():
    """Starts a fresh debate: clears memory and generates Agent A's opener."""
    data = request.json or {}
    topic = data.get("topic", "").strip()
    if not topic:
        return jsonify({"error": "A 'topic' field is required."}), 400

    conductor.reset()
    full_transcript["A"] = []
    full_transcript["B"] = []

    try:
        message = conductor.generate_agent_a_response(topic)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 502

    full_transcript["A"].append(message)

    return jsonify({
        "status": "active",
        "topic": topic,
        "agent": "A",
        "message": message,
    })


@app.route("/api/debate/next-turn", methods=["POST"])
def next_turn():
    """
    Generates the next agent's turn. Whichever agent DID NOT speak last
    goes next, mirroring the frontend's own turn logic.
    """
    data = request.json or {}
    topic = data.get("topic", "").strip()
    last_speaker = data.get("last_speaker")

    if not topic:
        return jsonify({"error": "A 'topic' field is required."}), 400

    next_agent = "B" if last_speaker == "A" else "A"

    try:
        if next_agent == "A":
            message = conductor.generate_agent_a_response(topic)
        else:
            message = conductor.generate_agent_b_response(topic)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 502

    full_transcript[next_agent].append(message)

    return jsonify({
        "agent": next_agent,
        "message": message,
    })


@app.route("/api/machine-learning/train", methods=["POST"])
def trigger_training():
    """Generates a mock dataset (if needed) and trains the regression judge."""
    try:
        dataset_path = "historical_debates.csv"
        import os
        if not os.path.exists(dataset_path):
            generate_mock_dataset(dataset_path)

        accuracy_metrics = ml_judge.train_model(dataset_path)
        return jsonify({"status": "Training Completed", "metrics": accuracy_metrics})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/machine-learning/evaluate", methods=["POST"])
def evaluate_debate():
    """
    Scores each agent's full combined argument text with the trained
    RandomForest judge and declares a winner.
    """
    data = request.json or {}
    advocate_text = data.get("advocate_text") or " ".join(full_transcript["A"])
    challenger_text = data.get("challenger_text") or " ".join(full_transcript["B"])

    if not ml_judge.is_trained:
        return jsonify({"error": "ML judge has not been trained yet. Call /api/machine-learning/train first."}), 400

    try:
        advocate_score = ml_judge.predict_score(advocate_text)
        challenger_score = ml_judge.predict_score(challenger_text)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    if advocate_score > challenger_score:
        winner = "Agent A (Advocate)"
    elif challenger_score > advocate_score:
        winner = "Agent B (Challenger)"
    else:
        winner = "Tie"

    return jsonify({
        "winner": winner,
        "advocate_score": advocate_score,
        "challenger_score": challenger_score,
    })


if __name__ == "__main__":
    print("🚀 AI Server running on http://127.0.0.1:5000")
    print("Ensure Ollama is running locally with 'qwen2.5:1.5b' pulled!")
    app.run(debug=True, port=5000)