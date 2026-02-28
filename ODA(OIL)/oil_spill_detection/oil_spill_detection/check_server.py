import urllib.request
import sys
try:
    with urllib.request.urlopen('http://127.0.0.1:5000', timeout=5) as r:
        print('STATUS', r.status)
        body = r.read(800).decode('utf-8', errors='replace')
        print('BODY PREVIEW:\n')
        print(body[:800])
except Exception as e:
    print('ERROR:', e)
    sys.exit(1)
