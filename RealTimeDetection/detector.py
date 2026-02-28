import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
from sklearn.decomposition import PCA

# Re-implementing the model architecture from the main project
class HamidaEtAl(nn.Module):
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

class RealTimeDetector:
    def __init__(self, model_path="file.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_channels = 34
        self.n_classes = 2
        self.patch_size = 3
        
        self.model = HamidaEtAl(self.input_channels, self.n_classes, self.patch_size).to(self.device)
        
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print(f"✅ Model loaded from {model_path}")
        else:
            print(f"⚠️ Model path {model_path} not found!")

    def detect(self, image_path, batch_size=2048):
        """Processes RGB image from GIBS and detects oil."""
        image = cv2.imread(image_path)
        if image is None: return None
        
        # 1. Convert to 34-band pseudo-hyperspectral data
        # GIBS provides RGB, we'll convert to grayscale and tile to use the existing model
        # Alternatively, we could use the 3 RGB bands and pad, but tiling grayscale is what app1.py does for JPG.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        pseudo_hs = np.tile(gray[:, :, np.newaxis], (1, 1, self.input_channels)).astype(np.float32)
        
        # 2. Reshape and normalize (transpose to C, H, W for the model's expected patch logic)
        full_image_tensor = pseudo_hs.transpose(2, 0, 1) # (34, H, W)
        
        # 3. Predict row-by-row (as in main project)
        pad = self.patch_size // 2
        padded_image = np.pad(full_image_tensor, ((0, 0), (pad, pad), (pad, pad)), mode='reflect')
        segmentation = np.zeros((h, w), dtype=np.uint8)
        
        with torch.no_grad():
            for i in range(h):
                row_patches = []
                for j in range(w):
                    patch = padded_image[:, i:i+self.patch_size, j:j+self.patch_size]
                    row_patches.append(patch)
                
                row_patches_np = np.stack(row_patches, axis=0)
                row_preds = []
                
                for b in range(0, w, batch_size):
                    chunk = row_patches_np[b:b+batch_size]
                    chunk_tensor = torch.from_numpy(chunk).float().unsqueeze(1).to(self.device)
                    output = self.model(chunk_tensor)
                    preds = torch.argmax(output, dim=1).cpu().numpy()
                    row_preds.append(preds)
                
                segmentation[i, :] = np.concatenate(row_preds)
                
        return segmentation

if __name__ == "__main__":
    # Test logic
    detector = RealTimeDetector()
    # Note: needs an image to test
