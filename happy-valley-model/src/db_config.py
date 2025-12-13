"""
Database Configuration Module

Provides a unified interface for database connections that supports both SQLite and PostgreSQL.
Set USE_POSTGRES=true in .env to use PostgreSQL, otherwise defaults to SQLite.

Usage:
    from src.db_config import get_connection
    
    conn = get_connection()
    cursor = conn.cursor()
    # ... database operations ...
    conn.close()
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Configuration
USE_POSTGRES = os.getenv('USE_POSTGRES', 'false').lower() == 'true'
DATABASE_URL = os.getenv('DATABASE_URL')
SQLITE_PATH = 'data/historical/hkjc.db'

def get_connection():
    """
    Get a database connection based on configuration.
    
    Returns:
        Database connection object
        - PostgreSQL: psycopg2 connection with RealDictCursor
        - SQLite: sqlite3 connection with Row factory
    """
    if USE_POSTGRES:
        import psycopg2
        from psycopg2.extras import RealDictCursor
        
        if not DATABASE_URL:
            raise ValueError("DATABASE_URL environment variable is required when USE_POSTGRES=true")
        
        return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    else:
        import sqlite3
        
        conn = sqlite3.connect(SQLITE_PATH)
        conn.row_factory = sqlite3.Row
        return conn

def get_db_type():
    """
    Get the current database type.
    
    Returns:
        str: 'postgresql' or 'sqlite'
    """
    return 'postgresql' if USE_POSTGRES else 'sqlite'

def get_placeholder():
    """
    Get the SQL parameter placeholder for the current database.
    
    Returns:
        str: '%s' for PostgreSQL, '?' for SQLite
    """
    return '%s' if USE_POSTGRES else '?'

