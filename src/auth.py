# src/auth.py
"""Streamlit authentication backed by the shared MySQL database layer."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

import bcrypt
import streamlit as st
from mysql.connector import IntegrityError

from src.database import (
    get_user_by_identifier,
    initialise_users_table,
    insert_user,
    update_user_last_login,
)


def _normalise_username(value: str) -> str:
    return value.strip()


def _normalise_email(value: str) -> str:
    return value.strip().lower()


def _validate_username(value: str) -> str | None:
    if len(value) < 3:
        return "Username must contain at least 3 characters."
    if len(value) > 30:
        return "Username must not exceed 30 characters."
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        return "Username may contain only letters, numbers, underscores, full stops and hyphens."
    return None


def _validate_email(value: str) -> str | None:
    if not re.fullmatch(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", value):
        return "Enter a valid email address."
    return None


def _validate_password(value: str) -> str | None:
    if len(value) < 8:
        return "Password must contain at least 8 characters."
    if not re.search(r"[A-Z]", value):
        return "Password must include at least one uppercase letter."
    if not re.search(r"[a-z]", value):
        return "Password must include at least one lowercase letter."
    if not re.search(r"\d", value):
        return "Password must include at least one number."
    return None


def create_user(username: str, email: str, password: str) -> tuple[bool, str]:
    initialise_users_table()
    username = _normalise_username(username)
    email = _normalise_email(email)

    error = (
        _validate_username(username)
        or _validate_email(email)
        or _validate_password(password)
    )
    if error:
        return False, error

    password_hash = bcrypt.hashpw(
        password.encode("utf-8"), bcrypt.gensalt()
    ).decode("utf-8")

    try:
        insert_user(
            username=username,
            email=email,
            password_hash=password_hash,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
    except IntegrityError:
        return False, "That username or email is already registered."

    return True, "Account created successfully. You can now sign in."


def authenticate_user(identifier: str, password: str) -> dict[str, Any] | None:
    initialise_users_table()
    row = get_user_by_identifier(identifier.strip())

    if row is None or not bool(row["is_active"]):
        return None

    if not bcrypt.checkpw(
        password.encode("utf-8"),
        str(row["password_hash"]).encode("utf-8"),
    ):
        return None

    last_login_at = datetime.now(timezone.utc).isoformat()
    update_user_last_login(int(row["id"]), last_login_at)

    return {
        "id": int(row["id"]),
        "username": row["username"],
        "email": row["email"],
        "created_at": row["created_at"],
        "last_login_at": last_login_at,
    }


def logout() -> None:
    st.session_state["authenticated"] = False
    st.session_state["user"] = None


def logout_button() -> None:
    if st.sidebar.button("Log out", use_container_width=True):
        logout()
        st.rerun()


def _login_form() -> None:
    st.subheader("Sign in")
    with st.form("login_form"):
        identifier = st.text_input("Username or email", autocomplete="username")
        password = st.text_input(
            "Password", type="password", autocomplete="current-password"
        )
        submitted = st.form_submit_button("Sign in", use_container_width=True)

    if submitted:
        if not identifier.strip() or not password:
            st.error("Enter your username or email and password.")
            return
        user = authenticate_user(identifier, password)
        if user is None:
            st.error("Invalid login details.")
            return
        st.session_state["authenticated"] = True
        st.session_state["user"] = user
        st.rerun()


def _registration_form() -> None:
    st.subheader("Create account")
    with st.form("registration_form"):
        username = st.text_input("Username", autocomplete="username")
        email = st.text_input("Email", autocomplete="email")
        password = st.text_input(
            "Password", type="password", autocomplete="new-password"
        )
        confirmation = st.text_input(
            "Confirm password", type="password", autocomplete="new-password"
        )
        submitted = st.form_submit_button("Create account", use_container_width=True)

    if submitted:
        if password != confirmation:
            st.error("Passwords do not match.")
            return
        created, message = create_user(username, email, password)
        st.success(message) if created else st.error(message)


def require_auth() -> dict[str, Any]:
    initialise_users_table()
    st.session_state.setdefault("authenticated", False)
    st.session_state.setdefault("user", None)

    if st.session_state["authenticated"] and st.session_state["user"]:
        return st.session_state["user"]

    st.title("Energy Efficiency Prediction")
    st.caption(
        "Sign in or create an account to access predictions and saved history."
    )
    login_tab, registration_tab = st.tabs(["Sign in", "Create account"])
    with login_tab:
        _login_form()
    with registration_tab:
        _registration_form()
    st.stop()
