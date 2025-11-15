from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Dict, List, Set, Tuple

import psycopg2
from psycopg2.extras import RealDictCursor


def _db_config() -> Dict[str, str]:
    return {
        "host": os.getenv("RAG_DB_HOST", os.getenv("POSTGRES_HOST", "localhost")),
        "port": os.getenv("RAG_DB_PORT", os.getenv("POSTGRES_PORT", "5432")),
        "dbname": os.getenv("RAG_DB_NAME", os.getenv("POSTGRES_DB", "postgres")),
        "user": os.getenv("RAG_DB_USER", os.getenv("POSTGRES_USER", "postgres")),
        "password": os.getenv("RAG_DB_PASSWORD", os.getenv("POSTGRES_PASSWORD", "postgres")),
        "connect_timeout": int(os.getenv("RAG_DB_CONNECT_TIMEOUT", "5")),
    }


@contextmanager
def get_connection():
    config = _db_config()
    conn = psycopg2.connect(**config)
    try:
        yield conn
    finally:
        conn.close()


def fetch_lessons_with_context() -> List[Dict[str, object]]:
    query = """
        SELECT
            lessons.id AS lesson_id,
            lessons.title AS lesson_title,
            lessons.content AS lesson_content,
            lessons.video_url AS lesson_video_url,
            lessons.file_url AS lesson_file_url,
            lessons.modified AS lesson_modified,
            lessons.course_id AS course_id,
            lessons.chapter_id AS chapter_id,
            chapters.title AS chapter_title,
            chapters.summary AS chapter_summary,
            chapters.modified AS chapter_modified,
            courses.title AS course_title,
            courses.short_introduction AS course_short_intro,
            courses.description AS course_description,
            courses.skill_level AS course_skill_level,
            courses.target_audience AS course_target_audience,
            courses.language AS course_language,
            courses.status AS course_status,
            courses.modified AS course_modified
        FROM lessons
        LEFT JOIN chapters ON chapters.id = lessons.chapter_id
        LEFT JOIN courses ON courses.id = lessons.course_id
        ORDER BY courses.title, chapters.position, lessons.position;
    """
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)
            return list(cur.fetchall())


def fetch_courses() -> List[Dict[str, object]]:
    query = """
        SELECT
            id AS course_id,
            title AS course_title,
            short_introduction AS course_short_intro,
            description AS course_description,
            target_audience AS course_target_audience,
            skill_level AS course_skill_level,
            language AS course_language,
            status AS course_status,
            modified AS course_modified
        FROM courses;
    """
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)
            return list(cur.fetchall())


def fetch_tags() -> Dict[Tuple[str, str], List[str]]:
    query = """
        SELECT entity_id, entity_type, array_agg(name ORDER BY name) AS names
        FROM tags
        GROUP BY entity_id, entity_type;
    """
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)
            rows = cur.fetchall()
    tags: Dict[Tuple[str, str], List[str]] = {}
    for row in rows:
        entity_id = str(row["entity_id"])
        entity_type = str(row["entity_type"])
        names = row["names"] or []
        tags[(entity_id, entity_type)] = [name for name in names if name]
    return tags


def fetch_labels() -> Dict[Tuple[str, str], List[str]]:
    query = """
        SELECT entity_id, entity_type, array_agg(name ORDER BY name) AS names
        FROM labels
        GROUP BY entity_id, entity_type;
    """
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)
            rows = cur.fetchall()
    labels: Dict[Tuple[str, str], List[str]] = {}
    for row in rows:
        entity_id = str(row["entity_id"])
        entity_type = str(row["entity_type"])
        names = row["names"] or []
        labels[(entity_id, entity_type)] = [name for name in names if name]
    return labels


def fetch_user_enrollments(user_id: str) -> Set[str]:
    query = """
        SELECT course_id
        FROM enrollments
        WHERE member_id = %s;
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (user_id,))
            rows = cur.fetchall()
    return {str(row[0]) for row in rows}

