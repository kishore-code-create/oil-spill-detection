#!/usr/bin/env python3
"""
Setup script to create the oil_spill_db database and tables.
"""
import pymysql

DB_HOST = os.environ.get('DB_HOST', 'localhost')
DB_USER = os.environ.get('DB_USER', 'root')
DB_PASSWORD = os.environ.get('DB_PASSWORD')
DB_PORT = 3306

# SQL commands to create database and tables
create_db_sql = "CREATE DATABASE IF NOT EXISTS oil_spill_db;"

create_users_table_sql = """
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(100) NOT NULL,
    password VARCHAR(100) NOT NULL
);
"""

create_history_table_sql = """
CREATE TABLE IF NOT EXISTS detection_history (
    id         INT AUTO_INCREMENT PRIMARY KEY,
    user_id    INT NOT NULL,
    username   VARCHAR(100) NOT NULL,
    method     VARCHAR(50) NOT NULL,
    filename   VARCHAR(255) NOT NULL,
    area_m2    FLOAT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
"""

try:
    # Connect to MySQL server (without specifying a database)
    conn = pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        port=DB_PORT
    )
    cursor = conn.cursor()
    
    # Create database
    cursor.execute(create_db_sql)
    print("✓ Database 'oil_spill_db' created or already exists.")

    cursor.execute("USE oil_spill_db;")

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
    
except pymysql.MySQLError as e:
    print(f"✗ Database error: {e}")
except Exception as e:
    print(f"✗ Error: {e}")
