from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime
from typing import Any, Dict, Iterator, List, Optional

import psycopg2
import psycopg2.extras

from config import settings


def _serialize_value(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def _serialize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {key: _serialize_value(value) for key, value in row.items()}


def _to_vector_literal(values: List[float]) -> str:
    return "[" + ",".join(f"{float(value):.12f}" for value in values) + "]"


@contextmanager
def get_conn() -> Iterator[psycopg2.extensions.connection]:
    conn = psycopg2.connect(
        host=settings.pg_host,
        port=settings.pg_port,
        dbname=settings.pg_database,
        user=settings.pg_user,
        password=settings.pg_password,
    )
    conn.set_client_encoding("UTF8")
    try:
        yield conn
    finally:
        conn.close()


def init_tables() -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS chats (
                    id SERIAL PRIMARY KEY,
                    title TEXT NOT NULL,
                    user_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id SERIAL PRIMARY KEY,
                    chat_id INT REFERENCES chats(id) ON DELETE CASCADE,
                    user_prompt TEXT NOT NULL,
                    rag_prompt TEXT NOT NULL,
                    generated_html TEXT NOT NULL,
                    preview_path TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            cur.execute(
                """
                ALTER TABLE chats
                ADD COLUMN IF NOT EXISTS user_id TEXT;
                """
            )
            cur.execute(
                """
                ALTER TABLE messages
                ADD COLUMN IF NOT EXISTS selected_image_url TEXT;
                """
            )
            conn.commit()


def create_chat(title: str, user_id: str) -> Dict[str, Any]:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                INSERT INTO chats (title, user_id)
                VALUES (%s, %s)
                RETURNING id, title, user_id, created_at;
                """,
                (title, user_id),
            )
            row = cur.fetchone()
            conn.commit()
            return _serialize_row(dict(row))


def get_chats(user_id: str) -> List[Dict[str, Any]]:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    c.id,
                    c.title,
                    c.user_id,
                    c.created_at,
                    COUNT(m.id) AS msg_count
                FROM chats c
                LEFT JOIN messages m ON m.chat_id = c.id
                WHERE c.user_id = %s
                GROUP BY c.id
                ORDER BY c.created_at DESC;
                """,
                (user_id,),
            )
            return [_serialize_row(dict(row)) for row in cur.fetchall()]


def chat_exists(chat_id: int, user_id: str) -> bool:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 1
                FROM chats
                WHERE id = %s AND user_id = %s;
                """,
                (chat_id, user_id),
            )
            return cur.fetchone() is not None


def get_messages(chat_id: int, user_id: str) -> List[Dict[str, Any]]:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    m.id,
                    m.user_prompt,
                    m.rag_prompt,
                    m.generated_html,
                    m.preview_path,
                    m.created_at,
                    m.selected_image_url
                FROM messages m
                JOIN chats c ON c.id = m.chat_id
                WHERE m.chat_id = %s AND c.user_id = %s
                ORDER BY m.created_at ASC;
                """,
                (chat_id, user_id),
            )
            return [_serialize_row(dict(row)) for row in cur.fetchall()]


def save_message(
    chat_id: int,
    user_prompt: str,
    rag_prompt: str,
    generated_html: str,
    preview_path: str,
    selected_image_url: Optional[str],
) -> Dict[str, Any]:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                INSERT INTO messages (
                    chat_id,
                    user_prompt,
                    rag_prompt,
                    generated_html,
                    preview_path,
                    selected_image_url
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING
                    id,
                    user_prompt,
                    rag_prompt,
                    generated_html,
                    preview_path,
                    created_at,
                    selected_image_url;
                """,
                (
                    chat_id,
                    user_prompt,
                    rag_prompt,
                    generated_html,
                    preview_path,
                    selected_image_url,
                ),
            )
            row = cur.fetchone()
            conn.commit()
            return _serialize_row(dict(row))


def delete_chat(chat_id: int, user_id: str) -> bool:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM chats WHERE id = %s AND user_id = %s;",
                (chat_id, user_id),
            )
            deleted = cur.rowcount > 0
            conn.commit()
            return deleted


def rename_chat(chat_id: int, title: str, user_id: str) -> bool:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE chats SET title = %s WHERE id = %s AND user_id = %s;",
                (title, chat_id, user_id),
            )
            updated = cur.rowcount > 0
            conn.commit()
            return updated


def fetch_image_by_url(image_url: str) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    id,
                    image_url,
                    description,
                    width,
                    height
                FROM ui_images
                WHERE image_url = %s
                LIMIT 1;
                """,
                (image_url,),
            )
            row = cur.fetchone()
            return dict(row) if row else None


def fetch_rag_context(query_embedding: List[float], limit_images: int = 3) -> Dict[str, Any]:
    vector_literal = _to_vector_literal(query_embedding)
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    page_type,
                    description,
                    ui_elements,
                    embedding <-> %s::vector AS distance
                FROM ui_page_structure
                ORDER BY embedding <-> %s::vector
                LIMIT 1;
                """,
                (vector_literal, vector_literal),
            )
            page_row = cur.fetchone()
            if not page_row:
                raise RuntimeError("Table ui_page_structure is empty.")
            page = dict(page_row)

            cur.execute(
                """
                SELECT
                    domain,
                    tone,
                    style_description,
                    embedding <-> %s::vector AS distance
                FROM ui_brand_style
                ORDER BY embedding <-> %s::vector
                LIMIT 1;
                """,
                (vector_literal, vector_literal),
            )
            brand_row = cur.fetchone()
            if not brand_row:
                raise RuntimeError("Table ui_brand_style is empty.")
            brand = dict(brand_row)

            cur.execute(
                """
                SELECT
                    id,
                    image_url,
                    description,
                    width,
                    height,
                    embedding <-> %s::vector AS distance
                FROM ui_images
                WHERE image_url IS NOT NULL
                ORDER BY embedding <-> %s::vector
                LIMIT %s;
                """,
                (vector_literal, vector_literal, limit_images),
            )
            images = [dict(row) for row in cur.fetchall()]

    return {"page": page, "brand": brand, "images": images}
