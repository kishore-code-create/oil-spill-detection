import requests
import os
from datetime import datetime, timedelta
from io import BytesIO
from PIL import Image

class GIBSService:
    """Service to interact with NASA GIBS WMS API."""
    
    BASE_URL = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
    
    def __init__(self, output_dir="downloads"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def fetch_latest_image(self, layer="MODIS_Terra_CorrectedReflectance_TrueColor", 
                           bbox="10,-90,30,-70", width=1024, height=512):
        """
        Fetches the latest available satellite image from GIBS.
        BBOX format: min_lat, min_lon, max_lat, max_lon
        Default BBOX: Gulf of Mexico
        """
        # GIBS usually has a slight delay for "latest", so try today then yesterday
        today = datetime.utcnow().strftime('%Y-%m-%d')
        yesterday = (datetime.utcnow() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        for date in [today, yesterday]:
            print(f"📡 Attempting to fetch {layer} for date: {date}...")
            params = {
                "SERVICE": "WMS",
                "VERSION": "1.3.0",
                "REQUEST": "GetMap",
                "LAYERS": layer,
                "CRS": "EPSG:4326",
                "BBOX": bbox,
                "WIDTH": str(width),
                "HEIGHT": str(height),
                "FORMAT": "image/png",
                "TIME": date
            }
            
            try:
                response = requests.get(self.BASE_URL, params=params, timeout=30)
                if response.status_code == 200 and len(response.content) > 1000: # Basic check for valid image
                    filename = f"gibs_{date}_{layer}.png"
                    path = os.path.join(self.output_dir, filename)
                    with open(path, "wb") as f:
                        f.write(response.content)
                    print(f"✅ Success! Image saved to {path}")
                    return path, date
                else:
                    print(f"⚠️ Data not yet available for {date} (Status: {response.status_code})")
            except Exception as e:
                print(f"❌ Error fetching from GIBS: {e}")
                
        return None, None

if __name__ == "__main__":
    # Test fetch
    service = GIBSService()
    path, date = service.fetch_latest_image()
    if path:
        print(f"Test Successful: {path} for {date}")
    else:
        print("Test Failed.")
