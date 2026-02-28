import os
import hashlib
import uuid
from datetime import datetime
from functools import wraps
import os

# Allow insecure transport for local development (OAuth2 requires HTTPS otherwise)
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

import pymysql
from flask import (Flask, render_template, request, redirect,
                   url_for, session, flash, g)
from werkzeug.utils import secure_filename
import shutil

import google.oauth2.id_token
from google_auth_oauthlib.flow import Flow
from google.auth.transport import requests as google_requests
from google.oauth2.credentials import Credentials
import googleapiclient.discovery
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import base64

app = Flask(__name__)
app.secret_key = 'portal_secret_key_oilspill_2025'

# Google OAuth2 Configuration
CLIENT_SECRETS_FILE = os.path.join(os.path.dirname(__file__), "..", "client_secret_*.json")
SCOPES = [
    "https://www.googleapis.com/auth/userinfo.profile", 
    "https://www.googleapis.com/auth/userinfo.email", 
    "https://www.googleapis.com/auth/gmail.send",
    "openid"
]

# ── DB config ──────────────────────────────────────────────────────────────────
DB_HOST     = os.environ.get('DB_HOST', 'localhost')
DB_USER     = os.environ.get('DB_USER', 'root')
DB_PASSWORD = os.environ.get('DB_PASSWORD', 'Nandu@2006')
DB_NAME     = 'oil_spill_portal_db'
DB_PORT     = 3306

UPLOAD_FOLDER  = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
ALLOWED_EXT    = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Role display names
ROLES = {
    'admin':                'Admin',
    'company_owner':        'Company Owner',
    'ngo':                  'NGO',
    'government_official':  'Government Official',
    'monitoring_system':    'Monitoring System',
    'coast_guard':          'Coast Guard',
    'environmental_agency': 'Environmental Agency',
    'media_press':          'Media / Press',
    'research_institution': 'Research Institution',
    'insurance_company':    'Insurance Company',
}

# Which roles see which severity levels (None = all)
ROLE_VISIBILITY = {
    'admin':                None,           # all
    'company_owner':        None,
    'government_official':  None,
    'monitoring_system':    None,
    'coast_guard':          ['High', 'Critical'],
    'environmental_agency': None,
    'ngo':                  None,
    'media_press':          ['Low', 'Medium', 'High', 'Critical'],  # summary only
    'research_institution': None,
    'insurance_company':    None,
}

# ── DB helpers ─────────────────────────────────────────────────────────────────
def get_db():
    if 'db' not in g:
        g.db = pymysql.connect(
            host=DB_HOST, user=DB_USER, password=DB_PASSWORD,
            database=DB_NAME, port=DB_PORT,
            cursorclass=pymysql.cursors.DictCursor,
            autocommit=False
        )
    return g.db

@app.teardown_appcontext
def close_db(exc):
    db = g.pop('db', None)
    if db: db.close()

def hash_pw(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

# ── Auth decorators ────────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in.', 'warning')
            return redirect(url_for('login', next=request.full_path))
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if session.get('role') != 'admin':
            flash('Admin access required.', 'danger')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route('/')
def home():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        uname = request.form.get('username', '').strip()
        pw    = request.form.get('password', '').strip()
        next_page = request.args.get('next') or request.form.get('next')
        
        db    = get_db()
        with db.cursor() as cur:
            cur.execute("SELECT * FROM portal_users WHERE username=%s AND is_active=1", (uname,))
            user = cur.fetchone()
        if user and user['password_hash'] == hash_pw(pw):
            session['user_id']  = user['id']
            session['username'] = user['username']
            session['role']     = user['role']
            session['full_name']= user['full_name'] or user['username']
            flash(f"Welcome, {session['full_name']}!", 'success')
            
            if next_page:
                return redirect(next_page)
            return redirect(url_for('admin_dash') if user['role'] == 'admin' else url_for('dashboard'))
        flash('Invalid credentials.', 'danger')
    return render_template('login.html', next=request.args.get('next'))

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully.', 'success')
    return redirect(url_for('login'))

# ── Google OAuth2 Routes ───────────────────────────────────────────────────────

@app.route('/login/google')
def login_google():
    next_page = request.args.get('next')
    flow = Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE,
        scopes=SCOPES,
        redirect_uri=url_for('oauth2callback', _external=True)
    )
    authorization_url, state = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true'
    )
    session['oauth_state'] = state
    session['code_verifier'] = flow.code_verifier
    session['oauth_next'] = next_page
    return redirect(authorization_url)

@app.route('/oauth2callback')
def oauth2callback():
    state = session.get('oauth_state')
    flow = Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE,
        scopes=SCOPES,
        state=state,
        redirect_uri=url_for('oauth2callback', _external=True)
    )
    flow.code_verifier = session.get('code_verifier')
    
    flow.fetch_token(authorization_response=request.url)
    
    credentials = flow.credentials
    id_info = google.oauth2.id_token.verify_oauth2_token(
        credentials.id_token, google_requests.Request(), credentials.client_id
    )
    
    google_id = id_info.get('sub')
    email = id_info.get('email')
    name = id_info.get('name')
    
    db = get_db()
    with db.cursor() as cur:
        # Try to find user by google_id or email
        cur.execute("SELECT * FROM portal_users WHERE google_id=%s OR email=%s", (google_id, email))
        user = cur.fetchone()
        
        if not user:
            flash(f'No account found with Gmail: {email}. Please contact admin to link your account.', 'danger')
            return redirect(url_for('login'))
        
        # If user exists but google_id not linked yet, link it
        if not user['google_id']:
            cur.execute("UPDATE portal_users SET google_id=%s WHERE id=%s", (google_id, user['id']))
            db.commit()
            
    # Success Login
    session['user_id'] = user['id']
    session['username'] = user['username']
    session['role'] = user['role']
    session['full_name'] = user['full_name'] or name or user['username']
    
    # Store credentials in session for notification usage
    creds_dict = {
        'token': credentials.token,
        'refresh_token': credentials.refresh_token,
        'token_uri': credentials.token_uri,
        'client_id': credentials.client_id,
        'client_secret': credentials.client_secret,
        'scopes': credentials.scopes
    }
    session['google_creds'] = creds_dict
    session['google_email'] = email
    
    next_page = session.pop('oauth_next', None)
    flash(f"Welcome, {session['full_name']}!", 'success')
    
    if next_page:
        return redirect(next_page)
    return redirect(url_for('admin_dash') if user['role'] == 'admin' else url_for('dashboard'))

# ── Email Notification Logic ──────────────────────────────────────────────────

def create_message(sender, to, subject, body_html):
    message = MIMEMultipart()
    message['to'] = to
    message['from'] = f"Oil Spill Portal <{sender}>"
    message['subject'] = subject
    
    msg_body = MIMEText(body_html, 'html')
    message.attach(msg_body)
    
    raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
    return {'raw': raw_message}

@app.route('/admin/test-gmail')
@login_required
@admin_required
def test_gmail():
    creds_dict = session.get('google_creds')
    if not creds_dict:
        flash('Connect to Google first.', 'warning')
        return redirect(url_for('admin_dash'))
        
    try:
        creds = Credentials(**creds_dict)
        service = googleapiclient.discovery.build('gmail', 'v1', credentials=creds)
        profile = service.users().getProfile(userId='me').execute()
        my_email = profile.get('emailAddress')
        
        body = f"<h1>Test Email</h1><p>This is a test from your Oil Spill Portal! If you see this, your Gmail integration is working.</p>"
        message = create_message(my_email, my_email, "Oil Spill Portal - TEST EMAIL", body)
        service.users().messages().send(userId='me', body=message).execute()
        
        flash(f'Success! A test email was sent to {my_email}. Check your inbox/spam.', 'success')
    except Exception as e:
        flash(f'Test failed: {str(e)}', 'danger')
        
    return redirect(url_for('admin_dash'))

@app.route('/report/<int:report_id>/notify', methods=['POST'])
@login_required
@admin_required
def notify_user_report(report_id):
    user_id = request.form.get('user_id')
    creds_dict = session.get('google_creds')
    
    if not creds_dict:
        flash('You must be logged in with Google to send Gmail notifications.', 'warning')
        return redirect(url_for('login_google', next=url_for('report_detail', report_id=report_id)))

    db = get_db()
    with db.cursor() as cur:
        cur.execute("SELECT full_name, email FROM portal_users WHERE id=%s", (user_id,))
        target_user = cur.fetchone()
        cur.execute("SELECT * FROM spill_reports WHERE id=%s", (report_id,))
        report = cur.fetchone()
        
    if not target_user or not target_user['email']:
        flash('Target user has no email address.', 'danger')
        return redirect(url_for('report_detail', report_id=report_id))
    
    try:
        creds = Credentials(**creds_dict)
        # Check if expired and refresh if possible (though we don't have a refresh token storage in DB yet, 
        # but the session dict should have it)
        if creds.expired and creds.refresh_token:
            creds.refresh(google_requests.Request())
            # Update the session with refreshed token
            refreshed_creds = {
                'token': creds.token,
                'refresh_token': creds.refresh_token,
                'token_uri': creds.token_uri,
                'client_id': creds.client_id,
                'client_secret': creds.client_secret,
                'scopes': creds.scopes
            }
            session['google_creds'] = refreshed_creds

        service = googleapiclient.discovery.build('gmail', 'v1', credentials=creds)
        
        # Get actual sender email for better headers
        profile = service.users().getProfile(userId='me').execute()
        sender_email = profile.get('emailAddress')
        
        email_body = f"""
        <html>
        <body style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; margin: 0; padding: 0;">
            <div style="max-width: 600px; margin: 20px auto; background: #ffffff; border: 1px solid #e1e4e8; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
                <div style="background: #0d1117; color: #ffffff; padding: 30px; text-align: center;">
                    <h1 style="margin: 0; color: #00aaff; font-size: 24px;">🚨 Oil Spill Alert</h1>
                    <p style="margin: 10px 0 0 0; opacity: 0.8; font-size: 14px;">Official Incident Notification</p>
                </div>
                <div style="padding: 30px;">
                    <p style="font-size: 16px;">Hello <b>{target_user['full_name'] or 'User'}</b>,</p>
                    <p style="font-size: 15px; color: #444;">This is a verified report regarding a detected oil spill that requires immediate review and potential response.</p>
                    
                    <div style="background: #f6f8fa; border-radius: 8px; padding: 20px; margin: 25px 0; border-left: 4px solid #00aaff;">
                        <table style="width: 100%; border-collapse: collapse;">
                            <tr><td style="padding: 5px 0; color: #666; font-size: 13px; width: 120px;">REPORT TITLE</td><td style="padding: 5px 0; font-weight: 600;">{report['title']}</td></tr>
                            <tr><td style="padding: 5px 0; color: #666; font-size: 13px;">LOCATION</td><td style="padding: 5px 0;">{report['location']}</td></tr>
                            <tr><td style="padding: 5px 0; color: #666; font-size: 13px;">SEVERITY</td><td style="padding: 5px 0;"><span style="color: {'#e74c3c' if report['severity'] in ['Critical', 'High'] else '#f1c40f'}; font-weight: 700;">{report['severity'].upper()}</span></td></tr>
                            <tr><td style="padding: 5px 0; color: #666; font-size: 13px;">EST. AREA</td><td style="padding: 5px 0;">{report['oil_area_m2']} m²</td></tr>
                        </table>
                    </div>
                    
                    <div style="text-align: center; margin: 35px 0;">
                        <a href="http://127.0.0.1:5001/report/{report_id}" 
                           style="background: #00aaff; color: #ffffff; padding: 14px 28px; text-decoration: none; border-radius: 8px; font-weight: bold; display: inline-block;">
                           ACCESS FULL REPORT
                        </a>
                    </div>
                </div>
                <div style="background: #f6f8fa; padding: 20px; text-align: center; border-top: 1px solid #e1e4e8;">
                    <p style="font-size: 12px; color: #888; margin: 0;">
                        Sent by {session.get('full_name')} via ODA Portal.<br>
                        This is an official communication related to environmental monitoring.
                    </p>
                </div>
            </div>
        </body>
        </html>
        """
        
        sub = f"ACTION REQUIRED: Oil Spill Report - {report['title']}"
        # Use actual sender email instead of 'me' for headers
        message = create_message(sender_email, target_user['email'], sub, email_body)
        
        print(f"DEBUG: Attempting to send email from {sender_email} to {target_user['email']}...")
        result = service.users().messages().send(userId='me', body=message).execute()
        print(f"DEBUG: Email success. ID: {result.get('id')}")
        
        flash(f"Notification successfully sent from {sender_email} to {target_user['email']}.", "success")
    except Exception as e:
        import traceback
        traceback.print_exc()
        flash(f"Gmail API Delivery Error: {str(e)}", "danger")
    
    return redirect(url_for('report_detail', report_id=report_id))

# ── User dashboard ─────────────────────────────────────────────────────────────
@app.route('/dashboard')
@login_required
def dashboard():
    role = session['role']
    if role == 'admin':
        return redirect(url_for('admin_dash'))

    db = get_db()
    with db.cursor() as cur:
        # Fetch reports visible to this role
        cur.execute("""
            SELECT r.*, u.full_name as author, u.organization
            FROM spill_reports r
            LEFT JOIN portal_users u ON r.created_by = u.id
            WHERE r.visible_to = 'all' OR r.visible_to LIKE %s
            ORDER BY r.created_at DESC
        """, (f'%{role}%',))
        reports = cur.fetchall()

        # Summary stats
        cur.execute("SELECT COUNT(*) as total FROM spill_reports")
        total = cur.fetchone()['total']
        cur.execute("SELECT COUNT(*) as active FROM spill_reports WHERE status='Active'")
        active = cur.fetchone()['active']
        cur.execute("SELECT COUNT(*) as critical FROM spill_reports WHERE severity='Critical'")
        critical = cur.fetchone()['critical']

    return render_template('dashboard.html',
                           reports=reports, role=role,
                           role_label=ROLES.get(role, role),
                           total=total, active=active, critical=critical)

# ── Report detail ──────────────────────────────────────────────────────────────
@app.route('/report/<int:report_id>')
@login_required
def report_detail(report_id):
    db = get_db()
    with db.cursor() as cur:
        cur.execute("""
            SELECT r.*, u.full_name as author, u.organization, u.role as author_role
            FROM spill_reports r
            LEFT JOIN portal_users u ON r.created_by = u.id
            WHERE r.id = %s
        """, (report_id,))
        report = cur.fetchone()
        
        if not report:
            flash('Report not found.', 'danger')
            return redirect(url_for('dashboard'))
            
        users = []
        if session.get('role') == 'admin':
            cur.execute("SELECT id, username, full_name, email FROM portal_users WHERE email IS NOT NULL AND email != ''")
            users = cur.fetchall()
            
            # Check if admin has Google credentials in session
            if not session.get('google_creds'):
                flash('Notice: You are logged in with a local account. To send Gmail notifications, please logout and use "Sign in with Google".', 'info')
            
    return render_template('report_detail.html', report=report, 
                           users=users,
                           role=session['role'], 
                           role_label=ROLES.get(session['role']))

# ── Admin routes ───────────────────────────────────────────────────────────────
@app.route('/admin')
@login_required
@admin_required
def admin_dash():
    db = get_db()
    with db.cursor() as cur:
        cur.execute("SELECT COUNT(*) as c FROM portal_users WHERE role != 'admin'")
        total_users = cur.fetchone()['c']
        cur.execute("SELECT COUNT(*) as c FROM spill_reports")
        total_reports = cur.fetchone()['c']
        cur.execute("SELECT COUNT(*) as c FROM spill_reports WHERE status='Active'")
        active_reports = cur.fetchone()['c']
        cur.execute("SELECT COUNT(*) as c FROM spill_reports WHERE severity='Critical'")
        critical_reports = cur.fetchone()['c']
        cur.execute("""
            SELECT r.*, u.full_name as author
            FROM spill_reports r LEFT JOIN portal_users u ON r.created_by=u.id
            ORDER BY r.created_at DESC LIMIT 10
        """)
        recent_reports = cur.fetchall()
    return render_template('admin/admin_dash.html',
                           total_users=total_users, total_reports=total_reports,
                           active_reports=active_reports, critical_reports=critical_reports,
                           recent_reports=recent_reports)

@app.route('/admin/users')
@login_required
@admin_required
def admin_users():
    db = get_db()
    with db.cursor() as cur:
        cur.execute("SELECT * FROM portal_users ORDER BY created_at DESC")
        users = cur.fetchall()
    return render_template('admin/users.html', users=users, ROLES=ROLES)

@app.route('/admin/users/create', methods=['GET', 'POST'])
@login_required
@admin_required
def create_user():
    if request.method == 'POST':
        uname = request.form.get('username', '').strip()
        pw    = request.form.get('password', '').strip()
        role  = request.form.get('role', 'ngo')
        fname = request.form.get('full_name', '').strip()
        email = request.form.get('email', '').strip()
        org   = request.form.get('organization', '').strip()
        if not uname or not pw:
            flash('Username and password are required.', 'danger')
            return render_template('admin/create_user.html', ROLES=ROLES)
        db = get_db()
        try:
            with db.cursor() as cur:
                cur.execute("""
                    INSERT INTO portal_users (username, password_hash, role, full_name, email, organization)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (uname, hash_pw(pw), role, fname, email, org))
            db.commit()
            flash(f'User "{uname}" created with role {ROLES[role]}.', 'success')
            return redirect(url_for('admin_users'))
        except pymysql.err.IntegrityError:
            flash(f'Username "{uname}" already exists.', 'danger')
    return render_template('admin/create_user.html', ROLES=ROLES)

@app.route('/admin/users/<int:uid>/toggle', methods=['POST'])
@login_required
@admin_required
def toggle_user(uid):
    db = get_db()
    with db.cursor() as cur:
        cur.execute("UPDATE portal_users SET is_active = NOT is_active WHERE id=%s", (uid,))
    db.commit()
    flash('User status updated.', 'success')
    return redirect(url_for('admin_users'))

@app.route('/admin/users/<int:uid>/delete', methods=['POST'])
@login_required
@admin_required
def delete_user(uid):
    db = get_db()
    with db.cursor() as cur:
        cur.execute("DELETE FROM portal_users WHERE id=%s AND role != 'admin'", (uid,))
    db.commit()
    flash('User deleted.', 'success')
    return redirect(url_for('admin_users'))

@app.route('/admin/reports')
@login_required
@admin_required
def admin_reports():
    db = get_db()
    with db.cursor() as cur:
        cur.execute("""
            SELECT r.*, u.full_name as author
            FROM spill_reports r LEFT JOIN portal_users u ON r.created_by=u.id
            ORDER BY r.created_at DESC
        """)
        reports = cur.fetchall()
    return render_template('admin/reports.html', reports=reports)

@app.route('/admin/reports/post', methods=['GET', 'POST'])
@login_required
@admin_required
def post_report():
    if request.method == 'POST':
        title    = request.form.get('title', '').strip()
        desc     = request.form.get('description', '').strip()
        location = request.form.get('location', '').strip()
        lat      = request.form.get('latitude') or None
        lon      = request.form.get('longitude') or None
        severity = request.form.get('severity', 'Medium')
        area     = request.form.get('oil_area_m2') or None
        volume   = request.form.get('estimated_volume') or None
        method   = request.form.get('detection_method', '').strip()
        status   = request.form.get('status', 'Active')
        visible  = ','.join(request.form.getlist('visible_to')) or 'all'

        image_path = None
        if 'image' in request.files:
            img = request.files['image']
            if img and img.filename and allowed_file(img.filename):
                fname = str(uuid.uuid4()) + '_' + secure_filename(img.filename)
                img.save(os.path.join(UPLOAD_FOLDER, fname))
                image_path = fname

        if not title:
            flash('Title is required.', 'danger')
            return render_template('admin/post_report.html', ROLES=ROLES)

        db = get_db()
        with db.cursor() as cur:
            cur.execute("""
                INSERT INTO spill_reports
                (title, description, location, latitude, longitude, severity,
                 oil_area_m2, estimated_volume, detection_method, image_path, created_by, status, visible_to)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (title, desc, location, lat, lon, severity,
                  area, volume, method, image_path, session['user_id'], status, visible))
        db.commit()
        flash('Report posted successfully!', 'success')
        return redirect(url_for('admin_reports'))

    # For GET requests, pre-fill from query params (for cross-app bridge)
    img_name = request.args.get('img', '')
    auto_submit = request.args.get('auto_submit') == 'true'
    image_path = None
    
    if img_name:
        # Try to copy from the detection app's history folder
        src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ODA(OIL)', 'oil_spill_detection', 'oil_spill_detection', 'static', 'history', img_name))
        if os.path.exists(src_path):
            new_fname = str(uuid.uuid4()) + "_" + img_name
            shutil.copy(src_path, os.path.join(UPLOAD_FOLDER, new_fname))
            image_path = new_fname

    prefill = {
        'title':    request.args.get('title', ''),
        'area':     request.args.get('area', ''),
        'volume':   request.args.get('volume', ''),
        'method':   request.args.get('method', ''),
        'location': request.args.get('location', ''),
        'image':    image_path,
        'severity': 'High' if (request.args.get('area') and float(request.args.get('area')) > 2000) else 'Medium'
    }

    if auto_submit and prefill['title']:
        db = get_db()
        with db.cursor() as cur:
            cur.execute("""
                INSERT INTO spill_reports
                (title, description, location, severity, oil_area_m2, estimated_volume, detection_method, image_path, created_by, status, visible_to)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (prefill['title'], "Automated report from Detection System.", prefill['location'], prefill['severity'], 
                  prefill['area'], prefill['volume'], prefill['method'], prefill['image'], session['user_id'], 'Active', 'all'))
        db.commit()
        report_id = cur.lastrowid
        flash('Report automatically posted successfully! Select recipient below to notify.', 'success')
        return redirect(url_for('report_detail', report_id=report_id))

    return render_template('admin/post_report.html', ROLES=ROLES, prefill=prefill)

@app.route('/admin/reports/<int:rid>/delete', methods=['POST'])
@login_required
@admin_required
def delete_report(rid):
    db = get_db()
    with db.cursor() as cur:
        cur.execute("SELECT image_path FROM spill_reports WHERE id=%s", (rid,))
        row = cur.fetchone()
        if row and row['image_path']:
            img_file = os.path.join(UPLOAD_FOLDER, row['image_path'])
            if os.path.exists(img_file):
                os.remove(img_file)
        cur.execute("DELETE FROM spill_reports WHERE id=%s", (rid,))
    db.commit()
    flash('Report deleted.', 'success')
    return redirect(url_for('admin_reports'))

@app.route('/admin/reports/<int:rid>/status', methods=['POST'])
@login_required
@admin_required
def update_report_status(rid):
    new_status = request.form.get('status', 'Active')
    db = get_db()
    with db.cursor() as cur:
        cur.execute("UPDATE spill_reports SET status=%s WHERE id=%s", (new_status, rid))
    db.commit()
    flash('Report status updated.', 'success')
    return redirect(url_for('admin_reports'))

if __name__ == '__main__':
    app.run(debug=True, port=5001, threaded=True)
