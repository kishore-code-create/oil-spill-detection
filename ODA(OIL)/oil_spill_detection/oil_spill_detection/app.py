# app.py
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
from flask_mysqldb import MySQL
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.io
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from tqdm import tqdm
from PIL import Image
from roboflow import Roboflow

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# MySQL Configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'Nandu@123'
app.config['MYSQL_DB'] = 'oil_spill_db'


mysql = MySQL(app)

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

def segment_full_image(model, image, patch_size, device):
    channels, H, W = image.shape
    pad = patch_size // 2
    padded_image = np.pad(image, ((0, 0), (pad, pad), (pad, pad)), mode='reflect')
    segmentation = np.zeros((H, W), dtype=np.int64)
    
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(H), desc="Segmenting rows"):
            for j in range(W):
                patch = padded_image[:, i:i+patch_size, j:j+patch_size]
                patch_tensor = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0).to(device)
                output = model(patch_tensor)
                pred = torch.argmax(output, dim=1).item()
                segmentation[i, j] = pred
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
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE username=%s AND password=%s", (username, password))
        user = cur.fetchone()
        cur.close()
        if user:
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid credentials. Please try again.', 'danger')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE username=%s", (username,))
        existing_user = cur.fetchone()
        if existing_user:
            flash('Username already exists. Please choose a different one.', 'danger')
        else:
            cur.execute("INSERT INTO users(username, password) VALUES(%s, %s)", (username, password))
            mysql.connection.commit()
            flash('Signup successful! You can now log in.', 'success')
            return redirect(url_for('login'))
        cur.close()
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
    
    # Example activities - in a real app, these would come from your database
    activities = [
        {
            'date': '2024-03-20',
            'method': 'Hyperspectral',
            'filename': 'sample1.mat',
            'status': 'Completed',
            'status_color': 'success'
        },
        {
            'date': '2024-03-19',
            'method': 'SAR',
            'filename': 'sample2.tiff',
            'status': 'Processing',
            'status_color': 'warning'
        }
    ]
    
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

            # Check if the uploaded file is a JPG/JPEG
            if file.filename.lower().endswith(('.jpg', '.jpeg')):
                try:
                    # Convert JPG to MAT file
                    image = cv2.imread(file_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    height, width, channels = image.shape
                    num_bands = 34
                    
                    # Expand the image to 34 bands
                    expanded_image = np.tile(image, (1, 1, num_bands // channels + 1))[:, :, :num_bands]
                    
                    # Save as MAT file
                    mat_filename = os.path.splitext(file.filename)[0] + '.mat'
                    mat_path = os.path.join(uploads_dir, mat_filename)
                    scipy.io.savemat(mat_path, {'img': expanded_image})
                    
                    flash('Image successfully converted to MAT format. Please proceed with the converted file.', 'success')
                    return redirect(request.url)
                
                except Exception as e:
                    flash(f'Error converting image: {str(e)}', 'danger')
                    return redirect(request.url)

            # Process MAT file
            elif file.filename.lower().endswith('.mat'):
                try:
                    # Load the model
                    input_channels = 34
                    n_classes = 2
                    patch_size = 3
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    model = HamidaEtAl(input_channels, n_classes, patch_size).to(device)
                    state_dict = torch.load("C:/Users/akira/codewithsenpa/iODA/2025_02_17_01_44_05_epoch60_0.98.pth", map_location=device)
                    model.load_state_dict(state_dict)

                    # Load and process the MAT file
                    mat_contents = scipy.io.loadmat(file_path)
                    if 'img' not in mat_contents:
                        flash("Variable 'img' not found in the .mat file.", 'danger')
                        return redirect(request.url)
                    
                    full_image = mat_contents['img']
                    H, W, C = full_image.shape

                    # Perform PCA
                    data_reshaped = full_image.reshape(-1, C)
                    pca = PCA(n_components=input_channels)
                    data_pca = pca.fit_transform(data_reshaped)
                    full_image_reduced = data_pca.reshape(H, W, input_channels).transpose(2, 0, 1)

                    # Perform segmentation
                    segmentation_result = segment_full_image(model, full_image_reduced, patch_size, device)

                    # Calculate area
                    total_area = calculate_area(segmentation_result)

                    # Save segmented image
                    segmented_image_path = os.path.join(uploads_dir, 'segmented_image.png')
                    plt.imsave(segmented_image_path, segmentation_result, cmap='jet')

                    # Create overlay image
                    overlay_image_path = os.path.join(uploads_dir, 'overlay_image.png')
                    rgb_image = full_image_reduced[:3, :, :].transpose(1, 2, 0)  # Convert to (H, W, 3)
                    rgb_image_norm = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min() + 1e-8)
                    plt.figure(figsize=(10, 8))
                    plt.imshow(rgb_image_norm)
                    plt.imshow(segmentation_result, cmap='jet', alpha=0.4)
                    plt.axis('off')
                    plt.savefig(overlay_image_path, bbox_inches='tight', pad_inches=0)
                    plt.close()

                    return render_template('volume_calculation.html', area=total_area, 
                                        segmented_image=segmented_image_path, 
                                        overlay_image=overlay_image_path)

                except Exception as e:
                    flash(f'Error processing MAT file: {str(e)}', 'danger')
                    return redirect(request.url)
            
            else:
                flash('Please upload either a JPG/JPEG image for conversion or a MAT file for processing.', 'warning')
                return redirect(request.url)

    return render_template('upload_hyperspectral.html')

@app.route('/calculate_volume', methods=['POST'])
def calculate_volume():
    area = float(request.form['area'])
    unit_choice = request.form['units']
    
    # Get thickness from form, default to 1 if not provided
    thickness_micrometers = float(request.form.get('thickness', 1))
    
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
    
    return render_template('results.html', volume_display=volume_display)

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
                # Initialize Roboflow (if not already initialized)
                rf = Roboflow(api_key="2YQXWnYr8GzfpiemnPRm")
                project = rf.workspace().project("oil-spill-yolo")
                model = project.version(1).model

                # Predict using the Roboflow model
                prediction = model.predict(file_path)
                output_path = os.path.join(outputs_dir, f"predicted_{file.filename}")
                prediction.save(output_path)

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

                # Save the side-by-side image
                side_by_side_filename = f"side_by_side_{file.filename}"
                side_by_side_path = os.path.join(outputs_dir, side_by_side_filename)
                side_by_side.save(side_by_side_path)

                # Pass the filename of the result image and the calculated area to the template
                return render_template('volume_calculation.html', area=oil_area, 
                                     segmented_image=side_by_side_path, 
                                     overlay_image=output_path)

            except Exception as e:
                flash(f'Error processing image: {str(e)}', 'danger')
                return redirect(request.url)

    return render_template('upload_sar.html')

if __name__ == '__main__':
    app.run(debug=True)