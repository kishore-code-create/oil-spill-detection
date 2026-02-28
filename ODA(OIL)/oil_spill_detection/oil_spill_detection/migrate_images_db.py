import pymysql
import os

# Detection App DB
DET_DB_NAME = 'oil_spill_db'
# Portal App DB
PORTAL_DB_NAME = 'oil_spill_portal_db'

DB_HOST = 'localhost'
DB_USER = 'root'
DB_PASSWORD = 'Nandu@2006'

def migrate():
    try:
        conn = pymysql.connect(host=DB_HOST, user=DB_USER, password=DB_PASSWORD)
        cur = conn.cursor()
        
        # 1. Update Detection App History
        cur.execute(f"USE {DET_DB_NAME}")
        cur.execute("SHOW COLUMNS FROM detection_history LIKE 'input_image'")
        if not cur.fetchone():
            print(f"Adding image columns to {DET_DB_NAME}.detection_history...")
            cur.execute("ALTER TABLE detection_history ADD COLUMN input_image VARCHAR(255) AFTER area_m2")
            cur.execute("ALTER TABLE detection_history ADD COLUMN output_image VARCHAR(255) AFTER input_image")
            print("Successfully added columns.")
        else:
            print(f"Image columns already exist in {DET_DB_NAME}.detection_history.")

        # 2. Update Portal App Reports (optional but good for consistency)
        cur.execute(f"USE {PORTAL_DB_NAME}")
        cur.execute("SHOW COLUMNS FROM spill_reports LIKE 'input_image_path'")
        if not cur.fetchone():
            print(f"Adding input_image_path column to {PORTAL_DB_NAME}.spill_reports...")
            cur.execute("ALTER TABLE spill_reports ADD COLUMN input_image_path VARCHAR(255) AFTER image_path")
            print("Successfully added column.")
        else:
            print(f"input_image_path column already exists in {PORTAL_DB_NAME}.spill_reports.")
            
        conn.commit()
        cur.close()
        conn.close()
        print("\nMigration completed successfully.")
    except Exception as e:
        print(f"Error during migration: {e}")

if __name__ == '__main__':
    migrate()
