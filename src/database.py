# src/database.py
"""
Shared MySQL database layer for the Energy Efficiency Prediction app.

This module manages:

- MySQL connection settings from .streamlit/secrets.toml
- User account storage
- Authentication-related user queries
- Prediction history storage
- Prediction history loading and deletion

The Streamlit interface design is not handled here.
This module contains database operations only.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st

try:
    import mysql.connector
    from mysql.connector import Error, IntegrityError
    from mysql.connector.connection import MySQLConnection
except ImportError as exc:
    raise ImportError(
        "MySQL support requires mysql-connector-python. "
        "Install it with:\n"
        "pip install mysql-connector-python"
    ) from exc


# DATABASE CONFIGURATION
# ---------------------------------------------------------------------
def get_database_config() -> dict[str, Any]:
    """
    Read MySQL credentials from Railway environment variables first.

    Local development falls back to:
        .streamlit/secrets.toml

    Railway variables:
        MYSQLHOST
        MYSQLPORT
        MYSQLDATABASE
        MYSQLUSER
        MYSQLPASSWORD
    """
    railway_host = os.getenv("MYSQLHOST", "").strip()

    if railway_host:
        config = {
            "host": railway_host,
            "port": int(os.getenv("MYSQLPORT", "3306")),
            "database": os.getenv("MYSQLDATABASE", "").strip(),
            "user": os.getenv("MYSQLUSER", "").strip(),
            "password": os.getenv("MYSQLPASSWORD", ""),
        }

        missing_keys = [
            key
            for key in ("host", "database", "user", "password")
            if str(config[key]).strip() == ""
        ]

        if missing_keys:
            raise RuntimeError(
                "Missing Railway MySQL environment variable(s): "
                + ", ".join(sorted(missing_keys))
            )

    else:
        try:
            mysql_secrets = st.secrets["mysql"]
        except Exception as exc:
            raise RuntimeError(
                "MySQL configuration was not found.\n\n"
                "For Railway, configure these environment variables:\n"
                "MYSQLHOST, MYSQLPORT, MYSQLDATABASE, MYSQLUSER, MYSQLPASSWORD\n\n"
                "For local development, create:\n"
                ".streamlit/secrets.toml\n\n"
                "Then add:\n\n"
                "[mysql]\n"
                'host = "localhost"\n'
                "port = 3306\n"
                'database = "energy_efficiency"\n'
                'user = "root"\n'
                'password = "your_mysql_password"'
            ) from exc

        required_keys = {
            "host",
            "database",
            "user",
            "password",
        }

        missing_keys = [
            key
            for key in required_keys
            if key not in mysql_secrets
            or str(mysql_secrets[key]).strip() == ""
        ]

        if missing_keys:
            raise RuntimeError(
                "Missing MySQL setting(s) in "
                ".streamlit/secrets.toml: "
                + ", ".join(sorted(missing_keys))
            )

        config = {
            "host": str(mysql_secrets["host"]),
            "port": int(mysql_secrets.get("port", 3306)),
            "database": str(mysql_secrets["database"]),
            "user": str(mysql_secrets["user"]),
            "password": str(mysql_secrets["password"]),
        }

    return {
        **config,
        "charset": "utf8mb4",
        "use_unicode": True,
        "autocommit": False,
        "connection_timeout": 10,
    }


def get_connection() -> MySQLConnection:
    """
    Open and return a MySQL connection.

    Returns
    -------
    MySQLConnection
        Active MySQL database connection.
    """
    config = get_database_config()

    try:
        connection = mysql.connector.connect(**config)
    except Error as exc:
        raise RuntimeError(
            "Could not connect to MySQL.\n\n"
            "Check that:\n"
            "1. The MySQL service is running.\n"
            "2. The database exists.\n"
            "3. The username and password are correct.\n"
            "4. Railway environment variables or local Streamlit secrets are correct.\n\n"
            f"MySQL error: {exc}"
        ) from exc

    if not connection.is_connected():
        raise RuntimeError(
            "MySQL returned a connection object, but the connection is not active."
        )

    return connection


# DATABASE INITIALISATION
# ---------------------------------------------------------------------
def initialise_users_table() -> None:
    """
    Create the users table and indexes when they do not exist.
    """
    connection = get_connection()
    cursor = connection.cursor()

    try:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INT UNSIGNED NOT NULL AUTO_INCREMENT,
                username VARCHAR(30) NOT NULL,
                email VARCHAR(255) NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                created_at VARCHAR(40) NOT NULL,
                last_login_at VARCHAR(40) NULL,
                is_active BOOLEAN NOT NULL DEFAULT TRUE,

                PRIMARY KEY (id),
                UNIQUE KEY uq_users_username (username),
                UNIQUE KEY uq_users_email (email),
                INDEX idx_users_username (username),
                INDEX idx_users_email (email)
            )
            ENGINE=InnoDB
            DEFAULT CHARACTER SET utf8mb4
            COLLATE utf8mb4_unicode_ci
            """
        )

        connection.commit()

    except Error:
        connection.rollback()
        raise

    finally:
        cursor.close()
        connection.close()


def initialise_predictions_table() -> None:
    """
    Create the predictions table and indexes when they do not exist.
    """
    connection = get_connection()
    cursor = connection.cursor()

    try:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
                user_id INT UNSIGNED NOT NULL,
                username VARCHAR(30) NOT NULL,
                created_at VARCHAR(40) NOT NULL,
                prediction DOUBLE NOT NULL,
                model_name VARCHAR(255) NULL,
                inputs_json JSON NOT NULL,

                PRIMARY KEY (id),

                INDEX idx_predictions_user_id (user_id),
                INDEX idx_predictions_created_at (created_at),

                CONSTRAINT fk_predictions_user
                    FOREIGN KEY (user_id)
                    REFERENCES users(id)
                    ON UPDATE CASCADE
                    ON DELETE CASCADE
            )
            ENGINE=InnoDB
            DEFAULT CHARACTER SET utf8mb4
            COLLATE utf8mb4_unicode_ci
            """
        )

        connection.commit()

    except Error:
        connection.rollback()
        raise

    finally:
        cursor.close()
        connection.close()


def initialise_database() -> None:
    """
    Initialise all tables required by the application.

    The users table must be created first because predictions.user_id
    references users.id.
    """
    initialise_users_table()
    initialise_predictions_table()


# USER DATABASE OPERATIONS
# ---------------------------------------------------------------------
def insert_user(
    username: str,
    email: str,
    password_hash: str,
    created_at: str,
) -> int:
    """
    Insert a new user account.

    Parameters
    ----------
    username:
        Normalised username.

    email:
        Normalised email address.

    password_hash:
        bcrypt password hash.

    created_at:
        ISO-formatted creation timestamp.

    Returns
    -------
    int
        ID of the newly created user.

    Raises
    ------
    IntegrityError
        When the username or email already exists.
    """
    initialise_users_table()

    connection = get_connection()
    cursor = connection.cursor()

    try:
        cursor.execute(
            """
            INSERT INTO users (
                username,
                email,
                password_hash,
                created_at,
                last_login_at,
                is_active
            )
            VALUES (%s, %s, %s, %s, NULL, TRUE)
            """,
            (
                username,
                email,
                password_hash,
                created_at,
            ),
        )

        user_id = int(cursor.lastrowid)
        connection.commit()

        return user_id

    except IntegrityError:
        connection.rollback()
        raise

    except Error:
        connection.rollback()
        raise

    finally:
        cursor.close()
        connection.close()


def get_user_by_identifier(
    identifier: str,
) -> dict[str, Any] | None:
    """
    Find a user by username or email.

    MySQL's utf8mb4_unicode_ci collation performs
    case-insensitive comparisons.
    """
    initialise_users_table()

    connection = get_connection()
    cursor = connection.cursor(dictionary=True)

    try:
        cursor.execute(
            """
            SELECT
                id,
                username,
                email,
                password_hash,
                created_at,
                last_login_at,
                is_active
            FROM users
            WHERE username = %s
               OR email = %s
            LIMIT 1
            """,
            (
                identifier,
                identifier.lower(),
            ),
        )

        row = cursor.fetchone()

        if row is None:
            return None

        return dict(row)

    finally:
        cursor.close()
        connection.close()


def update_user_last_login(
    user_id: int,
    last_login_at: str,
) -> None:
    """
    Save the user's most recent successful login time.
    """
    connection = get_connection()
    cursor = connection.cursor()

    try:
        cursor.execute(
            """
            UPDATE users
            SET last_login_at = %s
            WHERE id = %s
            """,
            (
                last_login_at,
                int(user_id),
            ),
        )

        connection.commit()

    except Error:
        connection.rollback()
        raise

    finally:
        cursor.close()
        connection.close()


def set_user_active_status(
    user_id: int,
    is_active: bool,
) -> None:
    """
    Activate or deactivate a user account.
    """
    connection = get_connection()
    cursor = connection.cursor()

    try:
        cursor.execute(
            """
            UPDATE users
            SET is_active = %s
            WHERE id = %s
            """,
            (
                bool(is_active),
                int(user_id),
            ),
        )

        connection.commit()

    except Error:
        connection.rollback()
        raise

    finally:
        cursor.close()
        connection.close()


# PREDICTION DATABASE OPERATIONS
# ---------------------------------------------------------------------
def save_prediction(
    user_id: int,
    username: str,
    prediction: float,
    model_name: str,
    inputs: dict[str, Any],
    created_at: str | None = None,
) -> int:
    """
    Save one prediction to MySQL.

    Parameters
    ----------
    user_id:
        ID of the authenticated user.

    username:
        Username of the authenticated user.

    prediction:
        Predicted heating-load value.

    model_name:
        Name of the model used.

    inputs:
        Original building parameters submitted by the user.

    created_at:
        Optional ISO timestamp. The current local time is used
        when this is not supplied.

    Returns
    -------
    int
        ID of the saved prediction.
    """
    initialise_database()

    timestamp = (
        created_at
        if created_at is not None
        else datetime.now().isoformat(timespec="seconds")
    )

    inputs_json = json.dumps(
        inputs,
        ensure_ascii=False,
    )

    connection = get_connection()
    cursor = connection.cursor()

    try:
        cursor.execute(
            """
            INSERT INTO predictions (
                user_id,
                username,
                created_at,
                prediction,
                model_name,
                inputs_json
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                int(user_id),
                str(username),
                timestamp,
                float(prediction),
                str(model_name),
                inputs_json,
            ),
        )

        prediction_id = int(cursor.lastrowid)
        connection.commit()

        return prediction_id

    except Error:
        connection.rollback()
        raise

    finally:
        cursor.close()
        connection.close()


def load_prediction_history(
    user_id: int,
    limit: int = 500,
) -> pd.DataFrame:
    """
    Load prediction history belonging to one user.

    Parameters
    ----------
    user_id:
        Authenticated user's database ID.

    limit:
        Maximum number of records to return.

    Returns
    -------
    pd.DataFrame
        Prediction history ordered from newest to oldest.
    """
    initialise_database()

    safe_limit = max(
        1,
        min(int(limit), 5000),
    )

    connection = get_connection()
    cursor = connection.cursor(dictionary=True)

    try:
        # LIMIT cannot always be passed safely as a normal parameter
        # across every MySQL connector version, so safe_limit is
        # converted to a validated integer above.
        query = f"""
            SELECT
                id,
                created_at,
                prediction,
                model_name,
                inputs_json
            FROM predictions
            WHERE user_id = %s
            ORDER BY id DESC
            LIMIT {safe_limit}
        """

        cursor.execute(
            query,
            (int(user_id),),
        )

        rows = cursor.fetchall()

        columns = [
            "id",
            "created_at",
            "prediction",
            "model_name",
            "inputs_json",
        ]

        if not rows:
            return pd.DataFrame(columns=columns)

        return pd.DataFrame(
            rows,
            columns=columns,
        )

    finally:
        cursor.close()
        connection.close()


def clear_prediction_history(
    user_id: int,
) -> int:
    """
    Delete all prediction history belonging to one user.

    Returns
    -------
    int
        Number of deleted records.
    """
    initialise_predictions_table()

    connection = get_connection()
    cursor = connection.cursor()

    try:
        cursor.execute(
            """
            DELETE FROM predictions
            WHERE user_id = %s
            """,
            (int(user_id),),
        )

        deleted_count = int(cursor.rowcount)
        connection.commit()

        return deleted_count

    except Error:
        connection.rollback()
        raise

    finally:
        cursor.close()
        connection.close()


def delete_prediction(
    prediction_id: int,
    user_id: int,
) -> bool:
    """
    Delete one prediction only when it belongs to the authenticated user.

    Returns
    -------
    bool
        True when a record was deleted.
    """
    connection = get_connection()
    cursor = connection.cursor()

    try:
        cursor.execute(
            """
            DELETE FROM predictions
            WHERE id = %s
              AND user_id = %s
            """,
            (
                int(prediction_id),
                int(user_id),
            ),
        )

        deleted = cursor.rowcount > 0
        connection.commit()

        return bool(deleted)

    except Error:
        connection.rollback()
        raise

    finally:
        cursor.close()
        connection.close()


# CONNECTION CHECK
# ---------------------------------------------------------------------
def test_connection() -> tuple[bool, str]:
    """
    Test the configured MySQL connection.

    Returns
    -------
    tuple[bool, str]
        Connection success status and readable message.
    """
    try:
        connection = get_connection()

        cursor = connection.cursor()
        cursor.execute("SELECT VERSION()")
        row = cursor.fetchone()

        version = str(row[0]) if row else "Unknown version"

        cursor.close()
        connection.close()

        return (
            True,
            f"MySQL connection successful. Server version: {version}",
        )

    except Exception as exc:
        return (
            False,
            str(exc),
        )
