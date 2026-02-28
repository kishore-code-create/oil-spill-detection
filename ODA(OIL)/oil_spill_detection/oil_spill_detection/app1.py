# app.py
import requests
import threading
import uuid
import json
import datetime
import time
import queue
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file, g, jsonify, Response, stream_with_context # backed logic
import pymysql
import numpy as np 
import cv2 #computer vision 
import matplotlib
matplotlib.use('Agg') # Set non-interactive backend for thread safety
import matplotlib.pyplot as plt
import scipy.io
import os 
import shutil
import torch
import torch.nn as nn #nndl(deep learning)
import torch.nn.functional as F
from sklearn.decomposition import PCA
from tqdm import tqdm
from PIL import Image
from roboflow import Roboflow

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET', 'your_secret_key')

# In-memory task store for tracking background processing progress
# { task_id: { 'progress': int, 'total': int, 'status': str, 'result': dict|None, 'error': str|None, 'queue': Queue } }
tasks = {}

# Database configuration (use environment variables when available)
DB_HOST = os.environ.get('DB_HOST', 'localhost')
DB_USER = os.environ.get('DB_USER', 'root')
DB_PASSWORD = os.environ.get('DB_PASSWORD', 'Nandu@2006')
DB_NAME = os.environ.get('DB_NAME', 'oil_spill_db')
DB_PORT = int(os.environ.get('DB_PORT', 3306))

# Model path (can be overridden by setting OIL_MODEL_PATH env var)
MODEL_DEFAULT_PATH = os.path.join(os.path.dirname(__file__), 'file.pth')
MODEL_PATH = os.environ.get('OIL_MODEL_PATH', MODEL_DEFAULT_PATH)

def get_db():
    """Get a database connection for the current request."""
    if 'db' not in g:
        g.db = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            port=DB_PORT,
            cursorclass=pymysql.cursors.DictCursor  # Optional: returns rows as dictionaries
        )
    return g.db

@app.teardown_appcontext
def close_db(exception):
    """Closes the database connection at the end of the request."""
    db = g.pop('db', None)
    if db is not None:
        db.close()

def save_detection(method, filename, area_m2, input_image=None, output_image=None, username=None):
    """Save a detection result to detection_history for the current user."""
    # Use provided username, or fallback to session
    effective_username = username or session.get('username')
    
    if not effective_username:
        print("Error saving detection: No username provided or found in session.")
        return
        
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM users WHERE username=%s", (effective_username,))
            user = cur.fetchone()
            if user:
                cur.execute(
                    "INSERT INTO detection_history (user_id, username, method, filename, area_m2, input_image, output_image) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (user['id'], effective_username, method, filename, area_m2, input_image, output_image)
                )
                conn.commit()
    except Exception as e:
        print(f"Error saving detection history: {e}")

# Define the model architecture
class HamidaEtAl(nn.Module):
    """
    3-D Deep Learning Approach for Remote Sensing Image Classification
    """
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=3, dilation=1):
        super(HamidaEtAl, self).__init__()
        self.patch_size = patch_size
        self.input_channels = input_channels
        dilation = (dilation, 1, 1)

        self.conv1 = nn.Conv3d(1, 20, (3, 3, 3), stride=(1, 1, 1), dilation=dilation, padding=1)
        self.pool1 = nn.Conv3d(20, 20, (3, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0))
        self.conv2 = nn.Conv3d(20, 35, (3, 3, 3), dilation=dilation, stride=(1, 1, 1), padding=(1, 0, 0))
        self.pool2 = nn.Conv3d(35, 35, (3, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0))
        self.conv3 = nn.Conv3d(35, 35, (3, 1, 1), dilation=dilation, stride=(1, 1, 1), padding=(1, 0, 0))
        self.conv4 = nn.Conv3d(35, 35, (2, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0))

        self.features_size = self._get_final_flattened_size()
        self.fc = nn.Linear(self.features_size, n_classes)
        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels, self.patch_size, self.patch_size))
            x = self.pool1(self.conv1(x))
            x = self.pool2(self.conv2(x))
            x = self.conv3(x)
            x = self.conv4(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, self.features_size)
        x = self.fc(x)
        return x

def segment_full_image(model, image, patch_size, device, progress_callback=None, batch_size=2048):
    channels, H, W = image.shape
    pad = patch_size // 2
    padded_image = np.pad(image, ((0, 0), (pad, pad), (pad, pad)), mode='reflect')
    segmentation = np.zeros((H, W), dtype=np.int64)
    
    model.eval()
    with torch.no_grad():
        for i in range(H):
            row_patches = []
            for j in range(W):
                patch = padded_image[:, i:i+patch_size, j:j+patch_size]
                row_patches.append(patch)
            
            row_patches_np = np.stack(row_patches, axis=0)
            row_preds = []
            
            # Process patches in batches to significantly speed up inference
            for b in range(0, W, batch_size):
                chunk = row_patches_np[b:b+batch_size]
                chunk_tensor = torch.from_numpy(chunk).float().unsqueeze(1).to(device)
                output = model(chunk_tensor)
                preds = torch.argmax(output, dim=1).cpu().numpy()
                row_preds.append(preds)
                
            row_result = np.concatenate(row_preds)
            segmentation[i, :] = row_result
            
            # Call progress callback after each row, passing the row data
            if progress_callback:
                progress_callback(i + 1, H, row_result.tolist())
    return segmentation

def calculate_area(segmented_img, pixel_width=3.3, pixel_height=3.3):
    object_pixels = np.count_nonzero(segmented_img)
    pixel_area = pixel_width * pixel_height
    total_area = object_pixels * pixel_area
    return total_area

@app.route('/')
def home():
    return render_template('base.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        try:
            conn = get_db()
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM users WHERE username=%s AND password=%s", (username, password))
                user = cur.fetchone()
            if user:
                session['username'] = username
                session['role'] = user.get('role', 'user')  # Default to 'user' if not set
                flash('Login successful!', 'success')
                return redirect(url_for('dashboard'))
            else:
                flash('Invalid credentials. Please try again.', 'danger')
        except Exception as e:
            flash(f"Database error: {str(e)}", "danger")
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        try:
            conn = get_db()
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM users WHERE username=%s", (username,))
                existing_user = cur.fetchone()
                if existing_user:
                    flash('Username already exists. Please choose a different one.', 'danger')
                else:
                    # New signups default to 'user' role
                    cur.execute("INSERT INTO users (username, password, role) VALUES(%s, %s, %s)", (username, password, 'user'))
                    conn.commit()
                    flash('Signup successful! You can now log in.', 'success')
                    return redirect(url_for('login'))
        except Exception as e:
            flash(f"Database error: {str(e)}", "danger")
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('home'))

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        flash('Please log in to access the dashboard.', 'warning')
        return redirect(url_for('login'))
    
    activities = []
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT method, filename, area_m2, input_image, output_image, created_at FROM detection_history WHERE username=%s ORDER BY created_at DESC",
                (session['username'],)
            )
            rows = cur.fetchall()
            for row in rows:
                activities.append({
                    'date': row['created_at'].strftime('%Y-%m-%d %H:%M') if row['created_at'] else 'N/A',
                    'method': row['method'],
                    'filename': row['filename'],
                    'area_m2': f"{row['area_m2']:.2f}" if row['area_m2'] is not None else 'N/A',
                    'input_image': row['input_image'],
                    'output_image': row['output_image'],
                    'status': 'Completed',
                    'status_color': 'success'
                })
    except Exception as e:
        flash(f'Could not load history: {e}', 'warning')
    
    return render_template('dashboard.html', activities=activities)

@app.route('/upload_hyperspectral', methods=['GET', 'POST'])
def upload_hyperspectral():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            uploads_dir = 'static/uploads'
            os.makedirs(uploads_dir, exist_ok=True)
            file_path = os.path.join(uploads_dir, file.filename)    
            file.save(file_path)

            # JPG → MAT conversion (fast, done inline)
            if file.filename.lower().endswith(('.jpg', '.jpeg')):
                try:
                    image = cv2.imread(file_path)
                    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    num_bands = 34
                    expanded_image = np.tile(gray_image[:, :, np.newaxis], (1, 1, num_bands))
                    mat_filename = os.path.splitext(file.filename)[0] + '.mat'
                    mat_path = os.path.join(uploads_dir, mat_filename)
                    scipy.io.savemat(mat_path, {'img': expanded_image})
                    session['converted_mat_file'] = mat_filename
                    flash(f'Image successfully converted to MAT format. You can <a href="{url_for("download_converted_file", filename=mat_filename)}">download the converted file</a> or upload it for processing.', 'success')
                    return redirect(request.url)
                except Exception as e:
                    flash(f'Error converting image: {str(e)}', 'danger')
                    return redirect(request.url)

            # MAT processing – run in background thread with SSE progress
            elif file.filename.lower().endswith('.mat'):
                task_id = str(uuid.uuid4())
                username = session.get('username')

                # Register the task
                tasks[task_id] = {
                    'progress': 0, 'total': 1,
                    'status': 'processing',
                    'result': None, 'error': None,
                    'q': queue.Queue(),
                    'start_time': time.time()
                }

                def run_processing(app_ctx, task_id, file_path, filename, username):
                    with app_ctx:
                        task = tasks[task_id]
                        try:
                            input_channels = 34
                            n_classes = 2
                            patch_size = 3
                            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                            model = HamidaEtAl(input_channels, n_classes, patch_size).to(device)

                            model_path = os.environ.get('OIL_MODEL_PATH', MODEL_PATH)
                            if not os.path.exists(model_path):
                                task['error'] = f'Model file not found at {model_path}.'
                                task['status'] = 'error'
                                task['q'].put(None)
                                return

                            state_dict = torch.load(model_path, map_location=device)
                            model.load_state_dict(state_dict)

                            mat_contents = scipy.io.loadmat(file_path)
                            if 'img' not in mat_contents:
                                task['error'] = "Variable 'img' not found in the .mat file."
                                task['status'] = 'error'
                                task['q'].put(None)
                                return

                            full_image = mat_contents['img']
                            H, W, C = full_image.shape
                            task['total'] = H
                            task['width'] = W
                            task['height'] = H

                            data_reshaped = full_image.reshape(-1, C)
                            pca = PCA(n_components=input_channels)
                            data_pca = pca.fit_transform(data_reshaped)
                            full_image_reduced = data_pca.reshape(H, W, input_channels).transpose(2, 0, 1)

                            def on_progress(current, total, row_data=None):
                                task['progress'] = current
                                task['total'] = total
                                if row_data is not None:
                                    task['latest_row'] = row_data
                                if 'pbar' in task:
                                    task['pbar'].update(1)

                            print(f"\n🚀 Starting Hyperspectral Processing for {filename}")
                            task['pbar'] = tqdm(total=H, desc="⏳ Segmenting Image (Batched)", unit="rows", leave=True)

                            segmentation_result = segment_full_image(
                                model, full_image_reduced, patch_size, device,
                                progress_callback=on_progress
                            )
                            
                            task['pbar'].close()
                            print("✅ Segmentation complete. Calculating stats and saving outputs...")

                            total_area = calculate_area(segmentation_result)

                            # Generate persistent history images
                            hist_id = str(uuid.uuid4())
                            hist_input_name = f"hyperspectral_input_{hist_id}.png"
                            hist_output_name = f"hyperspectral_output_{hist_id}.png"
                            
                            hist_input_path = os.path.join('static/history', hist_input_name)
                            hist_output_path = os.path.join('static/history', hist_output_name)
                            
                            # Create a visible RGB representation of the hyperspectral cube for history
                            # Assuming channels 29, 19, 9 roughly correspond to R, G, B in typical HS data
                            if C >= 30:
                                r_band = full_image[:, :, 29]
                                g_band = full_image[:, :, 19]
                                b_band = full_image[:, :, 9]
                            elif C >= 3:
                                r_band = full_image[:, :, 0]
                                g_band = full_image[:, :, 1]
                                b_band = full_image[:, :, 2]
                            else:
                                r_band = full_image[:, :, 0]
                                g_band = full_image[:, :, 0]
                                b_band = full_image[:, :, 0]

                            rgb_image = np.dstack((r_band, g_band, b_band))
                            
                            # Normalize for display
                            rgb_image_norm = cv2.normalize(rgb_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                            
                            # Save input (using the normalized RGB for visibility)
                            plt.figure(figsize=(10, 8))
                            plt.imshow(rgb_image_norm, cmap='gray')
                            plt.axis('off')
                            plt.savefig(hist_input_path, bbox_inches='tight', pad_inches=0)
                            plt.close()

                            # Save output (the overlay)
                            plt.figure(figsize=(10, 8))
                            plt.imshow(rgb_image_norm, cmap='gray')
                            plt.imshow(segmentation_result, cmap='gray', alpha=0.4)
                            plt.axis('off')
                            plt.savefig(hist_output_path, bbox_inches='tight', pad_inches=0)
                            plt.close()

                            # Save to DB using the helper, passing explicit username
                            save_detection('Hyperspectral', filename, total_area, hist_input_name, hist_output_name, username)

                            task['result'] = {
                                'area': total_area,
                                'segmented_image': hist_input_name,   # Input RGB
                                'overlay_image': hist_output_name,   # Result overlay
                                'history_image': hist_output_name
                            }
                            task['status'] = 'done'

                        except Exception as e:
                            task['error'] = str(e)
                            task['status'] = 'error'
                        finally:
                            task['q'].put(None)  # Signal completion

                t = threading.Thread(
                    target=run_processing,
                    args=(app.app_context(), task_id, file_path, file.filename, username),
                    daemon=True
                )
                t.start()

                return render_template('processing.html', task_id=task_id, filename=file.filename)
            else:
                flash('Please upload either a JPG/JPEG image for conversion or a MAT file for processing.', 'warning')
                return redirect(request.url)

    return render_template('upload_hyperspectral.html')

@app.route('/progress/<task_id>')
def progress(task_id):
    """SSE endpoint that streams segmentation progress for a given task."""
    def generate():
        task = tasks.get(task_id)
        if not task:
            yield f"data: {json.dumps({'error': 'Task not found'})}\'\n\n"
            return

        while task['status'] == 'processing':
            pct = int((task['progress'] / max(task['total'], 1)) * 100)
            elapsed = time.time() - task.get('start_time', time.time())
            
            remaining = 0
            speed = 0
            if task['progress'] > 0:
                speed = task['progress'] / max(elapsed, 0.001)
                remaining = (task['total'] - task['progress']) / speed if speed > 0 else 0
                
            def format_time(seconds):
                mins = int(seconds // 60)
                secs = int(seconds % 60)
                return f"{mins:02d}:{secs:02d}"

            data_payload = {
                'progress': task['progress'],
                'total': task['total'],
                'percent': pct,
                'status': 'processing',
                'elapsed': format_time(elapsed),
                'remaining': format_time(remaining),
                'speed': f"{speed:.2f}",
                'width': task.get('width'),
                'height': task.get('height')
            }
            
            # Include the latest row data if available
            if 'latest_row' in task:
                data_payload['latest_row'] = task['latest_row']
                # Clear it after sending to avoid sending the same row twice (optional, but cleaner)
                del task['latest_row']

            yield f"data: {json.dumps(data_payload)}\n\n"
            time.sleep(0.3) # Increased frequency slightly for smoother live view

        # Send final status
        if task['status'] == 'done':
            yield f"data: {json.dumps({'percent': 100, 'status': 'done'})}\n\n"
        else:
            yield f"data: {json.dumps({'status': 'error', 'error': task.get('error', 'Unknown error')})}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

@app.route('/task_result/<task_id>')
def task_result(task_id):
    """Returns the result of a completed task."""
    task = tasks.get(task_id)
    if not task:
        flash('Task not found.', 'danger')
        return redirect(url_for('upload_hyperspectral'))
    if task['status'] == 'done' and task['result']:
        result = task['result']
        # Cleanup task from memory
        tasks.pop(task_id, None)
        return render_template('volume_calculation.html',
                               area=result['area'],
                               segmented_image=result['segmented_image'],
                               overlay_image=result['overlay_image'],
                               history_image=result.get('history_image'),
                               method='Hyperspectral')
    elif task['status'] == 'error':
        flash(f"Processing error: {task.get('error', 'Unknown error')}", 'danger')
        tasks.pop(task_id, None)
        return redirect(url_for('upload_hyperspectral'))
    else:
        return redirect(url_for('progress_page', task_id=task_id))

@app.route('/processing/<task_id>')
def progress_page(task_id):
    task = tasks.get(task_id)
    if not task:
        flash('Task not found.', 'danger')
        return redirect(url_for('upload_hyperspectral'))
    return render_template('processing.html', task_id=task_id)

@app.route('/download_converted_file/<filename>')
def download_converted_file(filename):
    uploads_dir = 'static/uploads'
    file_path = os.path.join(uploads_dir, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        flash('File not found.', 'danger')
        return redirect(url_for('upload_hyperspectral'))

@app.route('/calculate_volume', methods=['POST'])
def calculate_volume():
    area = float(request.form['area'])
    unit_choice = request.form['units']
    
    # Get image filenames from form
    segmented_image = request.form.get('segmented_image')
    overlay_image = request.form.get('overlay_image')
    
    # Get thickness from form, default to 1 if not provided
    thickness_micrometers = float(request.form.get('thickness', 1))
    
    # Get method and history image for the bridge
    method = request.form.get('method', 'Detection')
    history_image = request.form.get('history_image')
    
    # Calculate volume using the provided thickness
    volume_cubic_meters = area * (thickness_micrometers * 1e-6)  # Convert μm to m

    if unit_choice == 'cubic_meters':
        volume = volume_cubic_meters
        volume_display = f"Estimated Volume: {volume:.6f} cubic meters"
    elif unit_choice == 'gallons':
        gallons_per_cubic_meter = 264.17
        volume = volume_cubic_meters * gallons_per_cubic_meter
        volume_display = f"Estimated Volume: {volume:.2f} gallons"
    elif unit_choice == 'both':
        gallons_per_cubic_meter = 264.17
        volume_gallons = volume_cubic_meters * gallons_per_cubic_meter
        volume_display = (f"Estimated Volume: {volume_cubic_meters:.6f} cubic meters<br>"
                          f"Estimated Volume: {volume_gallons:.2f} gallons")
    
    # Include thickness in the display
    thickness_info = f"<small>(Based on {thickness_micrometers} μm thickness)</small>"
    volume_display = f"{volume_display} {thickness_info}"
    
    return render_template('results.html', 
                          volume_display=volume_display, 
                          segmented_image=segmented_image,
                          overlay_image=overlay_image,
                          history_image=history_image,
                          area=area,
                          method=method,
                          now=datetime.datetime.now())

@app.route('/upload_sar', methods=['GET', 'POST'])
def upload_sar():
    if request.method == 'GET':
        return render_template('upload_sar.html')
    
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)

        if file:
            # Ensure the upload and output directories exist
            uploads_dir = os.path.join('static', 'uploads')
            outputs_dir = os.path.join('static', 'outputs')
            os.makedirs(uploads_dir, exist_ok=True)
            os.makedirs(outputs_dir, exist_ok=True)

            # Save the uploaded file
            file_path = os.path.join(uploads_dir, file.filename)
            file.save(file_path)

            try:
                print(f"\n📡 Starting SAR Analysis for {file.filename}")
                print("⏳ Connecting to Roboflow YOLOv8 Model...")
                # Initialize Roboflow (if not already initialized)
                rf = Roboflow(api_key="2YQXWnYr8GzfpiemnPRm")
                project = rf.workspace().project("oil-spill-yolo")
                model = project.version(1).model

                print("🔍 Requesting model inference...")
                # Predict using the Roboflow model
                prediction = model.predict(file_path)
                output_path = os.path.join(outputs_dir, f"predicted_{file.filename}")
                prediction.save(output_path)
                print("✅ Inference complete. Calculating severity and area...")

                # Calculate oil-affected area
                # Process the predicted image to get a binary mask of oil areas
                predicted_image = cv2.imread(output_path)
                
                # Convert to grayscale
                gray_image = cv2.cvtColor(predicted_image, cv2.COLOR_BGR2GRAY)
                
                # Apply Gaussian blur to reduce noise
                blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
                
                # Use adaptive thresholding instead of simple thresholding
                # This is more robust to different lighting conditions
                binary_mask = cv2.adaptiveThreshold(
                    blurred, 
                    255, 
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV, 
                    11, 
                    2
                )
                
                # Apply morphological operations to clean up the mask
                kernel = np.ones((3, 3), np.uint8)
                binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=2)
                binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
                
                # Count pixels representing oil
                oil_pixels = np.sum(binary_mask == 255)
                
                # Define pixel dimensions (in meters)
                pixel_size = 1.5 * 1.5  # Area of one pixel in m²
                
                # Calculate total area
                oil_area = oil_pixels * pixel_size

                # Create a side-by-side image
                original_image = Image.open(file_path)
                segmented_image = Image.open(output_path)

                # Ensure both images are the same size
                if original_image.size != segmented_image.size:
                    segmented_image = segmented_image.resize(original_image.size)

                # Create a new image with double the width
                side_by_side = Image.new('RGB', (original_image.width * 2, original_image.height))
                side_by_side.paste(original_image, (0, 0))
                side_by_side.paste(segmented_image, (original_image.width, 0))

                # Generate persistent history images
                hist_id = str(uuid.uuid4())
                hist_input_name = f"sar_input_{hist_id}.png"
                hist_output_name = f"sar_output_{hist_id}.png"
                
                hist_input_path = os.path.join('static/history', hist_input_name)
                hist_output_path = os.path.join('static/history', hist_output_name)
                
                # Copy original file to history
                shutil.copy(file_path, hist_input_path)
                # Save the side-by-side or results as output
                side_by_side.save(hist_output_path)

                # Save to detection history
                save_detection('SAR', file.filename, oil_area, hist_input_name, hist_output_name, session.get('username'))

                # Pass the filename of the result image and the calculated area to the template
                return render_template('volume_calculation.html', area=oil_area, 
                                     segmented_image=hist_input_name, 
                                     overlay_image=hist_output_name,
                                     history_image=hist_output_name,
                                     method='SAR')

            except Exception as e:
                flash(f'Error processing image: {str(e)}', 'danger')
                return redirect(request.url)

    return render_template('upload_sar.html')

# GROK API Chatbot Integration
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '')
        history = data.get('history', [])

        if not user_message:
            return jsonify({'error': 'Message cannot be empty'}), 400

        # System prompt restricting context to oil spill
        messages = [
            {"role": "system", "content": "You are a specialized AI assistant for the Oceanographic Data Analysts (ODA) Oil Spill Detection Platform. You MUST ONLY answer questions related to oil spills, marine environments, our detection software (Hyperspectral imaging and SAR imaging), calculating spill volumes, oceanography, or navigating this platform. If the user asks about anything completely unrelated (e.g., coding, general history, composing a poem, generic AI questions), you MUST politely decline and state you are only able to assist with topics related to oil spill detection and marine environments."}
        ]

        # Append previous history
        for msg in history:
            messages.append({"role": msg['role'], "content": msg['content']})

        # Append the new user message
        messages.append({"role": "user", "content": user_message})

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "llama-3.3-70b-versatile",  # Groq model
            "messages": messages,
            "temperature": 0.5,
            "max_tokens": 1024
        }

        response = requests.post(GROQ_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        reply = response.json()
        ai_message = reply['choices'][0]['message']['content']

        return jsonify({'message': ai_message})

    except requests.exceptions.RequestException as e:
        print(f"Groq API Error: {e}")
        return jsonify({'error': 'Failed to communicate with the Chatbot API.'}), 500
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': 'An internal server error occurred.'}), 500

if __name__ == '__main__':
    app.run(debug=True)

