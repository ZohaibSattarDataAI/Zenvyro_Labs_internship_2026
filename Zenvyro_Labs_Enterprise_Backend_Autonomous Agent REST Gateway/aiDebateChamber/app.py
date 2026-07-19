"""
app.py
------
Flask API gateway for the Autonomous AI Debate Chamber.

Routes:
    POST /api/debate/start                -> resets state, returns Agent A's opening statement
    POST /api/debate/next-turn            -> generates the next agent's turn (with memory)
    GET  /api/debate/history              -> returns full debate history
    POST /api/debate/reset                -> resets the debate state
    POST /api/machine-learning/train      -> trains the RandomForest judge on mock data
    POST /api/machine-learning/evaluate   -> scores both agents' full arguments and picks a winner

All heavy lifting lives in services/aiService.py (LangChain + Ollama calls) and
services/mlJudge.py (feature extraction + regression). This file is just
the routing/orchestration layer.
"""

import os
import logging
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from services.aiService import DebateConductor
from services.mlJudge import DebateRegressionJudge, generate_mock_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# The frontend (index.html, css/, js/) lives in ./frontend relative to this file.
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app)  # Allow frontend to call API from different origins

# System modules — single shared instances for the life of the server.
conductor = DebateConductor()
ml_judge = DebateRegressionJudge()

# Accumulates full-argument text per agent for the current debate, so the
# ML judge can evaluate the whole performance at the end, not just one turn.
full_transcript = {"A": [], "B": []}
current_topic = ""
debate_active = False


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


@app.route("/api/debate/start", methods=["POST"])
def start_debate():
    """
    Starts a fresh debate: clears memory and generates Agent A's opener.
    
    Request body:
        {
            "topic": "Will Artificial Intelligence Replace Human Jobs?"
        }
    
    Response:
        {
            "status": "active",
            "topic": "Will Artificial Intelligence Replace Human Jobs?",
            "agent": "A",
            "message": "I strongly support...",
            "turn": 1
        }
    """
    global current_topic, debate_active, full_transcript
    
    data = request.json or {}
    topic = data.get("topic", "").strip()
    
    if not topic:
        return jsonify({
            "error": "A 'topic' field is required.",
            "message": "Please provide a debate topic."
        }), 400
    
    # Reset state
    conductor.reset_history()
    full_transcript = {"A": [], "B": []}
    current_topic = topic
    debate_active = True
    
    try:
        message = conductor.generate_agent_a_response(topic)
        full_transcript["A"].append(message)
        turn_count = conductor.get_turn_count()
        
        logger.info(f"Debate started on topic: {topic}")
        
        return jsonify({
            "status": "active",
            "topic": topic,
            "agent": "A",
            "message": message,
            "turn": turn_count
        }), 200
        
    except RuntimeError as e:
        logger.error(f"Error starting debate: {str(e)}")
        return jsonify({
            "error": str(e),
            "message": "Failed to generate Agent A response."
        }), 503
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500


@app.route("/api/debate/next-turn", methods=["POST"])
def next_turn():
    """
    Generates the next agent's turn. Whichever agent DID NOT speak last
    goes next, mirroring the frontend's own turn logic.
    
    Request body:
        {
            "topic": "Will Artificial Intelligence Replace Human Jobs?",
            "last_speaker": "A"
        }
    
    Response:
        {
            "agent": "B",
            "message": "I strongly oppose...",
            "turn": 2
        }
    """
    global current_topic, debate_active, full_transcript
    
    if not debate_active:
        return jsonify({
            "error": "No active debate",
            "message": "Please start a debate first using /api/debate/start"
        }), 400
    
    data = request.json or {}
    topic = data.get("topic", "").strip() or current_topic
    last_speaker = data.get("last_speaker")
    
    if not topic:
        return jsonify({
            "error": "A 'topic' field is required.",
            "message": "Please provide the debate topic."
        }), 400
    
    # Determine next agent
    next_agent = "B" if last_speaker == "A" else "A"
    
    try:
        if next_agent == "A":
            message = conductor.generate_agent_a_response(topic)
        else:
            message = conductor.generate_agent_b_response(topic)
        
        full_transcript[next_agent].append(message)
        turn_count = conductor.get_turn_count()
        
        logger.info(f"Turn {turn_count}: Agent {next_agent} responded")
        
        return jsonify({
            "agent": next_agent,
            "message": message,
            "turn": turn_count
        }), 200
        
    except RuntimeError as e:
        logger.error(f"Error generating response: {str(e)}")
        return jsonify({
            "error": str(e),
            "message": f"Failed to generate Agent {next_agent} response."
        }), 503
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500


@app.route("/api/debate/history", methods=["GET"])
def get_history():
    """
    Get the full debate history.
    
    Response:
        {
            "history": [
                {
                    "speaker": "Agent A",
                    "text": "I strongly support...",
                    "turn": 1,
                    "timestamp": "2024-01-01T12:00:00"
                }
            ],
            "topic": "Will Artificial Intelligence Replace Human Jobs?",
            "active": true,
            "turn_count": 4
        }
    """
    global current_topic, debate_active, full_transcript
    
    return jsonify({
        "history": conductor.get_history(),
        "topic": current_topic,
        "active": debate_active,
        "turn_count": conductor.get_turn_count(),
        "transcript": full_transcript
    }), 200


@app.route("/api/debate/reset", methods=["POST"])
def reset_debate():
    """
    Reset the debate state.
    
    Response:
        {
            "status": "reset",
            "message": "Debate state has been reset"
        }
    """
    global current_topic, debate_active, full_transcript
    
    conductor.reset_history()
    full_transcript = {"A": [], "B": []}
    current_topic = ""
    debate_active = False
    
    logger.info("Debate reset")
    
    return jsonify({
        "status": "reset",
        "message": "Debate state has been reset"
    }), 200


@app.route("/api/machine-learning/train", methods=["POST"])
def trigger_training():
    """
    Generates a mock dataset (if needed) and trains the regression judge.
    
    Request body (optional):
        {
            "dataset_path": "historical_debates.csv",
            "force": false
        }
    
    Response:
        {
            "status": "Training Completed",
            "metrics": {
                "mse": 0.1234,
                "r2_score": 0.8567,
                "samples": 100,
                "features": 13
            }
        }
    """
    global ml_judge
    
    data = request.json or {}
    dataset_path = data.get("dataset_path", "historical_debates.csv")
    force = data.get("force", False)
    
    try:
        # Generate mock dataset if it doesn't exist or force is True
        if not os.path.exists(dataset_path) or force:
            generate_mock_dataset(dataset_path, num_samples=100, overwrite=force)
        
        # Train model
        metrics = ml_judge.train_model(dataset_path, save_model=True)
        
        logger.info(f"ML model trained successfully: {metrics}")
        
        return jsonify({
            "status": "Training Completed",
            "metrics": metrics
        }), 200
        
    except FileNotFoundError as e:
        logger.error(f"Dataset not found: {str(e)}")
        return jsonify({
            "error": "Dataset not found",
            "message": str(e)
        }), 404
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        return jsonify({
            "error": "Training failed",
            "message": str(e)
        }), 500


@app.route("/api/machine-learning/evaluate", methods=["POST"])
def evaluate_debate():
    """
    Scores each agent's full combined argument text with the trained
    RandomForest judge and declares a winner.
    
    Request body (optional):
        {
            "advocate_text": "Full Agent A argument...",
            "challenger_text": "Full Agent B argument..."
        }
    
    Response:
        {
            "winner": "Agent A (Advocate)",
            "advocate_score": 8.5,
            "challenger_score": 6.2,
            "advocate_features": {...},
            "challenger_features": {...}
        }
    """
    global full_transcript, ml_judge
    
    data = request.json or {}
    advocate_text = data.get("advocate_text") or " ".join(full_transcript["A"])
    challenger_text = data.get("challenger_text") or " ".join(full_transcript["B"])
    
    if not advocate_text or not challenger_text:
        return jsonify({
            "error": "No debate transcript available",
            "message": "Please start a debate first and generate at least one turn for each agent."
        }), 400
    
    if not ml_judge.is_trained:
        return jsonify({
            "error": "ML judge not trained",
            "message": "Please call /api/machine-learning/train first."
        }), 400
    
    try:
        # Get scores
        advocate_score = ml_judge.predict_score(advocate_text)
        challenger_score = ml_judge.predict_score(challenger_text)
        
        # Get features for transparency
        advocate_features = ml_judge.extract_nlp_features(advocate_text)
        challenger_features = ml_judge.extract_nlp_features(challenger_text)
        
        # Determine winner
        if advocate_score > challenger_score:
            winner = "Agent A (Advocate)"
        elif challenger_score > advocate_score:
            winner = "Agent B (Challenger)"
        else:
            winner = "Tie"
        
        logger.info(f"Evaluation complete - Winner: {winner}")
        
        return jsonify({
            "winner": winner,
            "advocate_score": advocate_score,
            "challenger_score": challenger_score,
            "advocate_features": advocate_features,
            "challenger_features": challenger_features,
            "advocate_length": len(advocate_text),
            "challenger_length": len(challenger_text)
        }), 200
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        return jsonify({
            "error": "Evaluation failed",
            "message": str(e)
        }), 500


@app.route("/api/health", methods=["GET"])
def health_check():
    """
    Health check endpoint.
    
    Response:
        {
            "status": "healthy",
            "ollama": "connected",
            "model": "qwen2.5:1.5b",
            "ml_model": "trained"
        }
    """
    ollama_status = "unknown"
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        ollama_status = "connected" if response.status_code == 200 else "error"
    except:
        ollama_status = "disconnected"
    
    return jsonify({
        "status": "healthy",
        "ollama": ollama_status,
        "model": conductor.model,
        "ml_model": "trained" if ml_judge.is_trained else "untrained",
        "debate_active": debate_active,
        "turn_count": conductor.get_turn_count()
    }), 200


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        "error": "Not Found",
        "message": "The requested endpoint does not exist"
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({
        "error": "Internal Server Error",
        "message": "An unexpected error occurred"
    }), 500


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 AI Debate Chamber Server")
    print("=" * 60)
    print(f"📁 Frontend directory: {FRONTEND_DIR}")
    print(f"🤖 Model: {conductor.model}")
    print(f"🔗 Ollama URL: {conductor.base_url}")
    print("=" * 60)
    print("🌐 Server running on: http://127.0.0.1:5000")
    print("📖 API Documentation available at /api/health")
    print("=" * 60)
    print("⚠️  Ensure Ollama is running locally with:")
    print("   ollama serve")
    print(f"   ollama pull {conductor.model}")
    print("=" * 60)
    
    app.run(debug=True, port=5000, host="127.0.0.1")