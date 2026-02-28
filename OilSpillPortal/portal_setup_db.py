import pymysql

import os
DB_HOST = os.environ.get('DB_HOST', 'localhost')
DB_USER = os.environ.get('DB_USER', 'root')
DB_PASSWORD = os.environ.get('DB_PASSWORD')
DB_PORT = 3306
DB_NAME = 'oil_spill_portal_db'

def setup():
    # Connect without selecting a DB first
    conn = pymysql.connect(host=DB_HOST, user=DB_USER, password=DB_PASSWORD, port=DB_PORT)
    cur = conn.cursor()

    cur.execute(f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
    cur.execute(f"USE `{DB_NAME}`")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS portal_users (
        id INT AUTO_INCREMENT PRIMARY KEY,
        username VARCHAR(100) NOT NULL UNIQUE,
        email VARCHAR(255),
        password_hash VARCHAR(255) NOT NULL,
        google_id VARCHAR(255),
        role ENUM(
            'admin',
            'company_owner',
            'ngo',
            'government_official',
            'monitoring_system',
            'coast_guard',
            'environmental_agency',
            'media_press',
            'research_institution',
            'insurance_company'
        ) NOT NULL DEFAULT 'ngo',
        full_name VARCHAR(200),
        organization VARCHAR(200),
        is_active BOOLEAN DEFAULT TRUE,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB;
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS spill_reports (
        id INT AUTO_INCREMENT PRIMARY KEY,
        title VARCHAR(255) NOT NULL,
        description TEXT,
        location VARCHAR(255),
        latitude FLOAT,
        longitude FLOAT,
        severity ENUM('Low', 'Medium', 'High', 'Critical') DEFAULT 'Medium',
        oil_area_m2 FLOAT,
        estimated_volume VARCHAR(255),
        detection_method VARCHAR(100),
        image_path VARCHAR(500),
        created_by INT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        status ENUM('Active', 'Contained', 'Resolved') DEFAULT 'Active',
        visible_to VARCHAR(500) DEFAULT 'all',
        FOREIGN KEY (created_by) REFERENCES portal_users(id) ON DELETE SET NULL
    ) ENGINE=InnoDB;
    """)

    conn.commit()

    # Create default admin account (password: admin123)
    import hashlib
    admin_pw = hashlib.sha256('admin123'.encode()).hexdigest()
    try:
        cur.execute("""
            INSERT INTO portal_users (username, password_hash, role, full_name, organization)
            VALUES ('admin', %s, 'admin', 'System Administrator', 'Oil Spill Monitoring Authority')
        """, (admin_pw,))
        conn.commit()
        print("✅ Default admin created: username=admin, password=admin123")
    except pymysql.err.IntegrityError:
        print("ℹ️  Admin account already exists.")

    cur.close()
    conn.close()
    print(f"✅ Database '{DB_NAME}' and tables created successfully.")

if __name__ == '__main__':
    setup()
