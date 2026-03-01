import streamlit as st
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io
import os
import matplotlib.pyplot as plt
from PIL import Image
from roboflow import Roboflow
from sklearn.decomposition import PCA
import requests
from datetime import datetime, timedelta
from io import BytesIO

# --- Model Architecture ---
class HamidaEtAl(nn.Module):
    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv3d)):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
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

# --- Helper Functions ---
def segment_full_image(model, image, patch_size, device):
    channels, H, W = image.shape
    pad = patch_size // 2
    padded_image = np.pad(image, ((0, 0), (pad, pad), (pad, pad)), mode='reflect')
    segmentation = np.zeros((H, W), dtype=np.int64)
    
    model.eval()
    with torch.no_grad():
        for i in range(H):
            patches = []
            for j in range(W):
                patch = padded_image[:, i:i+patch_size, j:j+patch_size]
                patches.append(patch)
            
            # Batching rows for speed if needed, but for simplicity:
            row_patches = np.stack(patches, axis=0)
            row_tensor = torch.from_numpy(row_patches).float().unsqueeze(1).to(device)
            output = model(row_tensor)
            preds = torch.argmax(output, dim=1).cpu().numpy()
            segmentation[i, :] = preds
    return segmentation

def calculate_area(segmented_img, pixel_width=3.3, pixel_height=3.3):
    object_pixels = np.count_nonzero(segmented_img)
    pixel_area = pixel_width * pixel_height
    total_area = object_pixels * pixel_area
    return total_area

class GIBSService:
    BASE_URL = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
    def fetch_latest_image(self, layer="MODIS_Terra_CorrectedReflectance_TrueColor", bbox="10,-90,30,-70"):
        today = datetime.utcnow().strftime('%Y-%m-%d')
        yesterday = (datetime.utcnow() - timedelta(days=1)).strftime('%Y-%m-%d')
        for date in [today, yesterday]:
            params = {
                "SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetMap",
                "LAYERS": layer, "CRS": "EPSG:4326", "BBOX": bbox,
                "WIDTH": "1024", "HEIGHT": "512", "FORMAT": "image/png", "TIME": date
            }
            try:
                response = requests.get(self.BASE_URL, params=params, timeout=30)
                if response.status_code == 200 and len(response.content) > 1000:
                    return response.content, date
            except: pass
        return None, None

# --- Custom CSS for Premium Design ---
st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at top right, #003366, #0E1117);
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 30px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    .main-title {
        background: linear-gradient(90deg, #00FFCC, #00CCFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3.5rem;
        margin-bottom: 0.5rem;
    }
    .stButton>button {
        background: linear-gradient(90deg, #00FFCC, #0099CC);
        color: #0E1117;
        font-weight: 600;
        border-radius: 12px;
        border: none;
        transition: all 0.3s ease;
        padding: 10px 25px;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 15px rgba(0,255,204,0.4);
    }
    .metric-card {
        background: rgba(0, 255, 204, 0.1);
        border: 1px solid rgba(0, 255, 204, 0.2);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('<h1 class="main-title">Oil Spill Detection Hub</h1>', unsafe_allow_html=True)
st.markdown("""
<div style="margin-bottom: 2rem; opacity: 0.8">
    Leveraging Neural Networks and Multi-Source Satellite Data for Real-Time Environmental Protection
</div>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.markdown("### 🛠️ CONTROL PANEL")
app_mode = st.sidebar.selectbox("Select Intelligence Core", ["Hyperspectral Analysis", "SAR Neural Detection", "NASA GIBS Live Monitor"])

# Shared Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.sidebar.divider()
st.sidebar.markdown(f"**Compute:** `{device.type.upper()}`")
if device.type == 'cpu':
    st.sidebar.warning("⚡ CPU Mode: Inference might be slow.")

# --- Main Logic ---
if app_mode == "Hyperspectral Analysis":
    st.subheader("🔬 Deep Hyperspectral Intelligence")
    
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.write("### 📤 Resource Upload")
        model_file = st.file_uploader("Intelligence Core Weights (.pth)", type=['pth'], help="Upload your pre-trained PyTorch weights.")
        uploaded_file = st.file_uploader("Satellite Data (JPG/MAT)", type=['jpg', 'jpeg', 'mat'], help="Upload high-res imagery or MAT data.")
        st.markdown('</div>', unsafe_allow_html=True)
        
    if uploaded_file and model_file:
        with st.spinner("🧠 Initializing Neural Core and Processing..."):
            # Load Model
            input_channels = 34
            n_classes = 2
            patch_size = 3
            model = HamidaEtAl(input_channels, n_classes, patch_size).to(device)
            model.load_state_dict(torch.load(model_file, map_location=device))
            
            # Load Image
            if uploaded_file.name.endswith('.mat'):
                mat_contents = scipy.io.loadmat(uploaded_file)
                full_image = mat_contents.get('img', np.zeros((10, 10, 34)))
            else:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, 1)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # Expand standard RGB/JPG to pseudo-hyperspectral
                full_image = np.tile(image, (1, 1, 34 // 3 + 1))[:, :, :34]
            
            # PCA for dimensionality mapping
            H, W, C = full_image.shape
            data_reshaped = full_image.reshape(-1, C)
            pca = PCA(n_components=input_channels)
            data_pca = pca.fit_transform(data_reshaped)
            full_image_reduced = data_pca.reshape(H, W, input_channels).transpose(2, 0, 1)
            
            # Neural Segmentation
            segmentation = segment_full_image(model, full_image_reduced, patch_size, device)
            area = calculate_area(segmentation)
            
            with col2:
                st.write("### 📊 Detection Analysis")
                tab1, tab2 = st.tabs(["Visualization", "Technical Specs"])
                
                with tab1:
                    fig, ax = plt.subplots(1, 2, figsize=(12, 6), facecolor='#0E1117')
                    ax[0].imshow(full_image[:, :, :3])
                    ax[0].set_title("Input RGB/Composite", color='white', pad=15)
                    ax[0].axis('off')
                    
                    # Custom colormap for better neon visibility
                    mask_colored = np.zeros((*segmentation.shape, 4))
                    mask_colored[segmentation == 1] = [0, 1, 0.8, 0.6] # Neon Cyan
                    
                    ax[1].imshow(full_image[:, :, :3])
                    ax[1].imshow(mask_colored)
                    ax[1].set_title("Detection Overlay", color='white', pad=15)
                    ax[1].axis('off')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
                with tab2:
                    st.json({
                        "Image Resolution": f"{W}x{H}",
                        "Spectral Bands": C,
                        "Model": "HamidaEtAl (3D-CNN)",
                        "Detected Pixels": int(np.count_nonzero(segmentation))
                    })

                st.divider()
                
                mcol1, mcol2 = st.columns(2)
                with mcol1:
                    st.markdown(f'<div class="metric-card"><h4>Affected Area</h4><h3>{area:,.2f} m²</h3></div>', unsafe_allow_html=True)
                
                with mcol2:
                    thickness = st.select_slider("Assumed Layer Thickness (μm)", options=[0.05, 0.1, 0.5, 1.0, 5.0, 10.0], value=1.0)
                    volume_m3 = area * (thickness * 1e-6)
                    st.markdown(f'<div class="metric-card"><h4>Est. Volume</h4><h3>{volume_m3:.6f} m³</h3></div>', unsafe_allow_html=True)
                
                st.info(f"💡 Equivalent to approximately **{volume_m3 * 264.172:,.2f} Gallons** of oil.")

    elif not model_file or not uploaded_file:
        st.info("Waiting for resource upload to begin analysis.")

elif app_mode == "SAR Neural Detection":
    st.subheader("📡 Synthetic Aperture Radar intelligence")
    st.write("Processing radar backscatter data via Roboflow API.")
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        api_key = st.text_input("Roboflow API Access Token", type="password", value="2YQXWnYr8GzfpiemnPRm")
        uploaded_file = st.file_uploader("Upload SAR Tiff/PNG", type=['jpg', 'jpeg', 'png', 'tiff'])
        st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file and api_key:
        with st.spinner("🛰️ Communicating with Roboflow Neural Edge..."):
            temp_path = f"tmp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            try:
                rf = Roboflow(api_key=api_key)
                project = rf.workspace().project("oil-spill-yolo")
                model = project.version(1).model
                
                prediction = model.predict(temp_path)
                pred_image_path = f"pred_{uploaded_file.name}.jpg"
                prediction.save(pred_image_path)
                
                with col2:
                    st.image(pred_image_path, use_container_width=True, caption="YOLO-V1 Neural Map")
                    
                    # Extract stats from prediction if available
                    boxes = prediction.json().get('predictions', [])
                    st.success(f"Successfully identified {len(boxes)} spill zones in SAR scan.")
                    
            except Exception as e:
                st.error(f"Execution Error: {e}")
            finally:
                if os.path.exists(temp_path): os.remove(temp_path)
                if 'pred_image_path' in locals() and os.path.exists(pred_image_path): os.remove(pred_image_path)

elif app_mode == "NASA GIBS Live Monitor":
    st.subheader("🌍 Planet-Scale Live Monitoring")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    scol1, scol2 = st.columns(2)
    with scol1:
        region = st.selectbox("Quick-Select Region", ["Gulf of Mexico", "North Sea", "Persian Gulf", "Custom BBOX"])
        bbox_map = {
            "Gulf of Mexico": "10,-90,30,-70",
            "North Sea": "50,-5,65,15",
            "Persian Gulf": "20,45,35,60",
            "Custom BBOX": "10,-90,30,-70"
        }
        bbox = st.text_input("Active BBOX (minLat,minLon,maxLat,maxLon)", bbox_map[region])
    
    with scol2:
        layer = st.selectbox("Satellite Layer", ["MODIS Terra True Color", "MODIS Aqua Corrected Reflectance", "Terra 721 Composite"])
        layer_map = {
            "MODIS Terra True Color": "MODIS_Terra_CorrectedReflectance_TrueColor",
            "MODIS Aqua Corrected Reflectance": "MODIS_Aqua_CorrectedReflectance_TrueColor",
            "Terra 721 Composite": "MODIS_Terra_CorrectedReflectance_Bands721"
        }
        
    if st.button("📡 Synchronize with Satellite Hub"):
        gibs = GIBSService()
        with st.spinner("Establishing Satellite Uplink..."):
            img_content, date = gibs.fetch_latest_image(layer=layer_map[layer], bbox=bbox)
            
            if img_content:
                st.success(f"Live Sync Complete. Timestamp: {date}")
                st.image(img_content, caption=f"NASA GIBS Acquisition - {date}", use_container_width=True)
                st.info("💡 You can download this image and upload it to the 'Hyperspectral Analysis' tab for neural processing.")
            else:
                st.error("Uplink failed. Region may have high cloud cover or technical downtime. Please adjust BBOX.")
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; opacity: 0.6">
    Developed for Advanced Maritime Environmental Monitoring | Powered by PyTorch & Streamlit
</div>
""", unsafe_allow_html=True)
