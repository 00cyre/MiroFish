"""
Setup API — save/load the Anthropic key (API key or Claude OAuth token).

POST /api/setup/key   { "key": "sk-ant-..." }   → saves to ~/.mirofish/config.json
GET  /api/setup/key                              → returns { "configured": true/false, "type": "oauth"|"api"|null }
"""

from flask import Blueprint, jsonify, request

from ..utils.anthropic_client import load_anthropic_key, save_anthropic_key

setup_bp = Blueprint("setup", __name__, url_prefix="/api/setup")


@setup_bp.route("/key", methods=["GET"])
def get_key_status():
    key = load_anthropic_key()
    if not key:
        return jsonify({"configured": False, "type": None})
    key_type = "oauth" if key.startswith("sk-ant-oat") else "api"
    return jsonify({"configured": True, "type": key_type})


@setup_bp.route("/key", methods=["POST"])
def save_key():
    data = request.get_json(silent=True) or {}
    key = data.get("key", "").strip()
    if not key:
        return jsonify({"error": "key is required"}), 400
    if not key.startswith("sk-ant-"):
        return jsonify({"error": "key must start with sk-ant-"}), 400
    save_anthropic_key(key)
    key_type = "oauth" if key.startswith("sk-ant-oat") else "api"
    return jsonify({"ok": True, "type": key_type})
