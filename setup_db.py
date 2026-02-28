#!/usr/bin/env python3
"""
Setup script to create the oil_spill_db tables in Postgres.
"""
import os
import psycopg2

DB_HOST = os.environ.get('DB_HOST', 'localhost')
DB_USER = os.environ.get('DB_USER', 'postgres')
DB_PASSWORD = os.environ.get('DB_PASSWORD')
DB_PORT = os.environ.get('DB_PORT', 5432)
DB_NAME = os.environ.get('DB_NAME', 'oil_spill_db')

# SQL commands to create tables (Postgres syntax)
create_users_table_sql = """
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) NOT NULL UNIQUE,
    password VARCHAR(100) NOT NULL
);
"""

create_history_table_sql = """
CREATE TABLE IF NOT EXISTS detection_history (
    id          SERIAL PRIMARY KEY,
    user_id     INT NOT NULL,
    username    VARCHAR(100) NOT NULL,
    method      VARCHAR(50) NOT NULL,
    filename    VARCHAR(255) NOT NULL,
    area_m2     FLOAT,
    input_image VARCHAR(255),
    output_image VARCHAR(255),
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_user FOREIGN KEY (user_id) REFERENCES users(id)
);
"""

try:
    print(f"Connecting to Postgres at {DB_HOST}:{DB_PORT}...")
    conn = psycopg2.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        port=DB_PORT,
        database=DB_NAME
    )
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute(create_users_table_sql)
    print("✓ Table 'users' created or already exists.")
    
    # Create detection_history table
    cursor.execute(create_history_table_sql)
    print("✓ Table 'detection_history' created or already exists.")
    
    # Commit and close
    conn.commit()
    cursor.close()
    conn.close()
    
    print("\n✓ Database setup completed successfully!")
    
except Exception as e:
    print(f"✗ Error: {e}")
