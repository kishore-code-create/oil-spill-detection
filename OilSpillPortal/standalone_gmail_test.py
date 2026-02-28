import os
import base64
import json
from flask import Flask, request, redirect
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
import threading
import time

# This script will run on port 5001 to match your authorized redirect URI
PORT = 5001
CLIENT_SECRETS_FILE = os.path.join(os.path.dirname(__file__), "..", "client_secret_*.json")
SCOPES = ['https://www.googleapis.com/auth/gmail.send', 'https://www.googleapis.com/auth/userinfo.email', 'openid']

app = Flask(__name__)
# Allow insecure transport for local testing
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

@app.route('/')
def index():
    try:
        global flow
        flow = Flow.from_client_secrets_file(
            CLIENT_SECRETS_FILE,
            scopes=SCOPES,
            redirect_uri=f'http://127.0.0.1:{PORT}/oauth2callback'
        )
        auth_url, _ = flow.authorization_url(prompt='consent')
        return f'<h1>Gmail Diagnostic</h1><p>Click below to authorize:</p><a href="{auth_url}">Authorize Gmail API</a>'
    except Exception as e:
        import traceback
        return f"<h1>Error during Auth Init</h1><pre>{traceback.format_exc()}</pre>"

@app.route('/oauth2callback')
def oauth2callback():
    try:
        flow.fetch_token(authorization_response=request.url)
        creds = flow.credentials
        
        service = build('gmail', 'v1', credentials=creds)
        user_info = service.users().getProfile(userId='me').execute()
        my_email = user_info.get('emailAddress')
        
        recipient = "nandakishoredevarashetti@gmail.com"
        subject = "DIAGNOSTIC TEST: Gmail API (Flask-based)"
        body = f"Hello! This is a diagnostic test from {my_email}. If you received this, the API is working perfectly."

        message = MIMEMultipart()
        message['to'] = recipient
        message['from'] = f"Diagnostic <{my_email}>"
        message['subject'] = subject
        message.attach(MIMEText(body, 'html'))

        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        
        sent = service.users().messages().send(userId='me', body={'raw': raw}).execute()
        result = f"""
        <h2 style='color: green;'>✅ SUCCESS!</h2>
        <p>Authenticated as: <b>{my_email}</b></p>
        <p>Message sent to: <b>{recipient}</b></p>
        <p>Message ID: {sent['id']}</p>
        <p>Check your inbox and spam folder now.</p>
        """
        # Simple trick to shut down the server after a few seconds
        threading.Thread(target=lambda: (time.sleep(5), os._exit(0))).start()
        return result
    except Exception as e:
        import traceback
        return f"<h1>Error during Callback/Send</h1><pre>{traceback.format_exc()}</pre>"

if __name__ == '__main__':
    print(f"Starting diagnostic server on http://127.0.0.1:{PORT}")
    print("Please open this URL in your browser.")
    app.run(port=PORT, host='127.0.0.1', debug=True, use_reloader=False)
