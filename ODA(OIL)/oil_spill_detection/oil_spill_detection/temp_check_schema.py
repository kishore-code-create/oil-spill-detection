import pymysql
import os

DB_HOST = 'localhost'
DB_USER = 'root'
DB_PASSWORD = 'Nandu@2006'
DB_NAME = 'oil_spill_db'

def check_schema():
    try:
        conn = pymysql.connect(host=DB_HOST, user=DB_USER, password=DB_PASSWORD, database=DB_NAME, cursorclass=pymysql.cursors.DictCursor)
        cur = conn.cursor()
        cur.execute("DESCRIBE detection_history")
        rows = cur.fetchall()
        print("Schema for detection_history:")
        for row in rows:
            print(f"{row['Field']}: {row['Type']}")
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    check_schema()
