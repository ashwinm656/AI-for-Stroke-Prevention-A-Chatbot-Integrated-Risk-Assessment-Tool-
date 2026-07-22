"""
Tiny local auth + assessment-history store for NeuroCare.

Uses SQLite (stdlib) so there are no extra dependencies to deploy.
NOTE: on Streamlit Community Cloud the filesystem is ephemeral — the
neurocare.db file will be wiped on redeploys / when the app sleeps and
wakes up. Fine for demoing this locally or for a single running session;
for real persistence, swap this for a hosted DB (e.g. Supabase/Postgres)
and keep the same function signatures.
"""
import hashlib
import json
import os
import sqlite3
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "neurocare.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    conn.execute(
        """CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at TEXT NOT NULL
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            age INTEGER NOT NULL,
            gender TEXT NOT NULL,
            symptoms_json TEXT NOT NULL,
            at_risk INTEGER NOT NULL,
            risk_pct INTEGER NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )"""
    )
    conn.commit()
    conn.close()


def _hash_password(password, salt=None):
    if salt is None:
        salt = os.urandom(16).hex()
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 100_000).hex()
    return digest, salt


def create_user(username, email, password):
    """Returns (user_dict, None) on success, (None, error_message) on failure."""
    username = username.strip()
    email = email.strip().lower()
    if not username or not email or not password:
        return None, "All fields are required."
    if len(password) < 6:
        return None, "Password must be at least 6 characters."

    conn = get_conn()
    existing = conn.execute(
        "SELECT id FROM users WHERE username = ? OR email = ?", (username, email)
    ).fetchone()
    if existing:
        conn.close()
        return None, "That username or email is already registered."

    pw_hash, salt = _hash_password(password)
    cur = conn.execute(
        "INSERT INTO users (username, email, password_hash, salt, created_at) VALUES (?, ?, ?, ?, ?)",
        (username, email, pw_hash, salt, datetime.utcnow().isoformat()),
    )
    conn.commit()
    user_id = cur.lastrowid
    conn.close()
    return {"id": user_id, "username": username, "email": email}, None


def verify_user(username_or_email, password):
    """Returns user_dict on success, None on failure."""
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM users WHERE username = ? OR email = ?",
        (username_or_email.strip(), username_or_email.strip().lower()),
    ).fetchone()
    conn.close()
    if not row:
        return None
    pw_hash, _ = _hash_password(password, row["salt"])
    if pw_hash != row["password_hash"]:
        return None
    return {"id": row["id"], "username": row["username"], "email": row["email"]}


def save_assessment(user_id, age, gender, symptom_labels, symptom_keys, at_risk, risk_pct):
    conn = get_conn()
    conn.execute(
        "INSERT INTO assessments (user_id, created_at, age, gender, symptoms_json, at_risk, risk_pct) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            user_id, datetime.utcnow().isoformat(), age, gender,
            json.dumps({"labels": symptom_labels, "keys": symptom_keys}),
            int(at_risk), int(risk_pct),
        ),
    )
    conn.commit()
    conn.close()


def get_latest_assessment(user_id):
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM assessments WHERE user_id = ? ORDER BY created_at DESC LIMIT 1", (user_id,)
    ).fetchone()
    conn.close()
    if not row:
        return None
    symptoms = json.loads(row["symptoms_json"])
    return {
        "created_at": row["created_at"], "age": row["age"], "gender": row["gender"],
        "symptoms": symptoms["labels"], "symptom_keys": symptoms["keys"],
        "at_risk": bool(row["at_risk"]), "risk_pct": row["risk_pct"],
    }


def get_history(user_id, limit=10):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM assessments WHERE user_id = ? ORDER BY created_at DESC LIMIT ?", (user_id, limit)
    ).fetchall()
    conn.close()
    out = []
    for row in rows:
        symptoms = json.loads(row["symptoms_json"])
        out.append({
            "created_at": row["created_at"], "age": row["age"], "gender": row["gender"],
            "symptoms": symptoms["labels"], "at_risk": bool(row["at_risk"]), "risk_pct": row["risk_pct"],
        })
    return out


def get_stats(user_id):
    conn = get_conn()
    row = conn.execute(
        "SELECT COUNT(*) as cnt, MAX(created_at) as last FROM assessments WHERE user_id = ?", (user_id,)
    ).fetchone()
    conn.close()
    return {"count": row["cnt"], "last": row["last"]}
