from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import uuid
import cv2
import numpy as np
import matplotlib.pyplot as plt
from gibs_service import GIBSService
from detector import RealTimeDetector

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/results'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

gibs = GIBSService(output_dir='static/downloads')
detector = RealTimeDetector(model_path='file.pth')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/fetch_and_detect', methods=['POST'])
def fetch_and_detect():
    # 1. Fetch from GIBS
    # Defaulting to Gulf of Mexico as per plan
    img_path, date = gibs.fetch_latest_image()
    
    if not img_path:
        return jsonify({'error': 'Could not fetch image from NASA GIBS. Please try again later.'}), 500
    
    # 2. Run detection
    mask = detector.detect(img_path)
    if mask is None:
        return jsonify({'error': 'Detection failed.'}), 500
    
    # 3. Create visualization
    original = cv2.imread(img_path)
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # Overlay mask on original
    overlay = original_rgb.copy()
    overlay[mask == 1] = [0, 255, 204] # Neon Cyan
    
    # Save the result
    result_id = str(uuid.uuid4())
    result_filename = f"result_{result_id}.png"
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
    
    # Use matplotlib to save a side-by-side comparison
    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.imshow(original_rgb)
    plt.title(f"Original Satellite View ({date})")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.title("Detected Oil Spill Zones")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(result_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    return jsonify({
        'status': 'success',
        'date': date,
        'original_url': f"/static/downloads/{os.path.basename(img_path)}",
        'result_url': f"/static/results/{result_filename}"
    })

if __name__ == '__main__':
    app.run(port=5001, debug=True)
