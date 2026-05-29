from __future__ import annotations

import os
import uuid

from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS

from config import settings
from db import (
    chat_exists,
    create_chat,
    delete_chat,
    get_chats,
    get_messages,
    init_tables,
    rename_chat,
    save_message,
)
from rag import generate_html, suggest_image_candidates

app = Flask(__name__)
CORS(app)

os.makedirs(settings.preview_dir, exist_ok=True)
init_tables()


def get_request_user_id() -> str:
    user_id = (request.headers.get("X-User-Id") or "").strip()
    if not user_id:
        raise ValueError("Missing X-User-Id header.")
    try:
        uuid.UUID(user_id)
    except ValueError as exc:
        raise ValueError("Invalid X-User-Id header.") from exc
    return user_id


@app.errorhandler(ValueError)
def handle_value_error(exc: ValueError):
    return jsonify({"error": str(exc)}), 400


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/preview/<path:filename>")
def serve_preview(filename: str):
    return send_from_directory(settings.preview_dir, filename)


@app.route("/api/chats", methods=["GET"])
def api_get_chats():
    user_id = get_request_user_id()
    return jsonify(get_chats(user_id))


@app.route("/api/chats", methods=["POST"])
def api_create_chat():
    user_id = get_request_user_id()
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "New Chat").strip() or "New Chat"
    return jsonify(create_chat(title, user_id))


@app.route("/api/chats/<int:chat_id>", methods=["DELETE"])
def api_delete_chat(chat_id: int):
    user_id = get_request_user_id()
    if not delete_chat(chat_id, user_id):
        return jsonify({"error": "Chat not found."}), 404
    return jsonify({"ok": True})


@app.route("/api/chats/<int:chat_id>", methods=["PATCH"])
def api_rename_chat(chat_id: int):
    user_id = get_request_user_id()
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "Chat").strip() or "Chat"
    if not rename_chat(chat_id, title, user_id):
        return jsonify({"error": "Chat not found."}), 404
    return jsonify({"ok": True})


@app.route("/api/chats/<int:chat_id>/messages", methods=["GET"])
def api_get_messages(chat_id: int):
    user_id = get_request_user_id()
    if not chat_exists(chat_id, user_id):
        return jsonify({"error": "Chat not found."}), 404
    return jsonify(get_messages(chat_id, user_id))


@app.route("/api/image-options", methods=["POST"])
def api_image_options():
    get_request_user_id()
    data = request.get_json(silent=True) or {}
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "Prompt is required."}), 400

    try:
        result = suggest_image_candidates(prompt)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify(result)


@app.route("/api/chats/<int:chat_id>/generate", methods=["POST"])
def api_generate(chat_id: int):
    user_id = get_request_user_id()
    if not chat_exists(chat_id, user_id):
        return jsonify({"error": "Chat not found."}), 404

    data = request.get_json(silent=True) or {}
    user_prompt = (data.get("prompt") or "").strip()
    selected_image_url = (data.get("selected_image_url") or "").strip() or None

    if not user_prompt:
        return jsonify({"error": "Prompt is required."}), 400

    try:
        result = generate_html(user_prompt=user_prompt, selected_image_url=selected_image_url)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    filename = f"{uuid.uuid4().hex}.html"
    file_path = os.path.join(settings.preview_dir, filename)
    with open(file_path, "w", encoding="utf-8") as file_obj:
        file_obj.write(result["html"])

    preview_url = f"/preview/{filename}"
    saved_message = save_message(
        chat_id=chat_id,
        user_prompt=user_prompt,
        rag_prompt=result["rag_prompt"],
        generated_html=result["html"],
        preview_path=preview_url,
        selected_image_url=result["selected_image"]["image_url"] if result["selected_image"] else None,
    )

    saved_message["image_candidates"] = result["image_candidates"]
    saved_message["selected_image"] = result["selected_image"]
    return jsonify(saved_message)


if __name__ == "__main__":
    app.run(host=settings.app_host, port=settings.app_port, debug=settings.debug)
