import pymysql
import os

DB_HOST = 'localhost'
DB_USER = 'root'
DB_PASSWORD = 'Nandu@2006'
DB_NAME = 'oil_spill_db'

def migrate():
    try:
        conn = pymysql.connect(host=DB_HOST, user=DB_USER, password=DB_PASSWORD, database=DB_NAME)
        cur = conn.cursor()
        
        # Check if role column exists
        cur.execute("SHOW COLUMNS FROM users LIKE 'role'")
        result = cur.fetchone()
        
        if not result:
            print("Adding 'role' column to 'users' table...")
            cur.execute("ALTER TABLE users ADD COLUMN role VARCHAR(20) DEFAULT 'user'")
            # Set current users to admin for testing convenience (or leave as user)
            cur.execute("UPDATE users SET role='admin' WHERE username='admin'")
            conn.commit()
            print("Successfully added 'role' column.")
        else:
            print("'role' column already exists.")
            
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error during migration: {e}")

if __name__ == '__main__':
    migrate()
