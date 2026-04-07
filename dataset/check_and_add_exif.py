#!/usr/bin/env python3
"""Check existing EXIF and add fake EXIF data to images without it."""

import os
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import random
from datetime import datetime, timedelta

DATASET_DIR = "."

# Landmark GPS coordinates (latitude, longitude)
LANDMARKS = {
    "b6.jpg": (29.9792, 31.1342),  # Pyramids of Giza, Egypt
    "bund.jpg": (31.2304, 121.4737),  # Shanghai Bund, China
    "syd5.jpg": (-33.8568, 151.2153),  # Sydney Opera House, Australia
    "tower.jpg": (48.8584, 2.2945),  # Eiffel Tower, Paris, France
    "天安门.jpg": (39.9055, 116.3976),  # Tiananmen, Beijing, China
}

# Random locations for generic photos (latitude, longitude)
# Mix of beach locations, city locations, and random spots
RANDOM_LOCATIONS = [
    (25.7617, -80.1918),   # Miami Beach, USA
    (36.3932, 25.4615),    # Santorini, Greece
    (-8.4095, 115.1889),   # Bali, Indonesia
    (35.0116, 135.7681),   # Kyoto, Japan
    (41.9028, 12.4964),    # Rome, Italy
    (51.5074, -0.1278),    # London, UK
    (40.7128, -74.0060),   # New York, USA
    (34.0522, -118.2437),  # Los Angeles, USA
    (55.7558, 37.6173),    # Moscow, Russia
    (-23.5505, -46.6333),  # São Paulo, Brazil
    (1.3521, 103.8198),    # Singapore
    (13.7563, 100.5018),   # Bangkok, Thailand
    (37.5665, 126.9780),   # Seoul, South Korea
    (28.6139, 77.2090),    # New Delhi, India
    (52.5200, 13.4050),    # Berlin, Germany
]

def get_existing_exif(image_path):
    """Check if image already has EXIF data with timestamp and GPS."""
    try:
        img = Image.open(image_path)
        exif = img._getexif()
        
        if not exif:
            return None, None
        
        timestamp = None
        gps_info = None
        
        for tag_id, value in exif.items():
            tag = TAGS.get(tag_id, tag_id)
            
            if tag == "DateTimeOriginal" or tag == "DateTime":
                timestamp = value
            
            if tag == "GPSInfo":
                gps_info = value
        
        return timestamp, gps_info
    except Exception as e:
        print(f"Error reading {image_path}: {e}")
        return None, None


def decimal_to_dms(decimal):
    """Convert decimal degrees to degrees, minutes, seconds."""
    degrees = int(decimal)
    minutes = int((abs(decimal) - abs(degrees)) * 60)
    seconds = round(((abs(decimal) - abs(degrees)) * 60 - minutes) * 60, 6)
    return (degrees, minutes, seconds)


def create_gps_exif(lat, lon):
    """Create GPS EXIF data structure."""
    lat_dms = decimal_to_dms(abs(lat))
    lon_dms = decimal_to_dms(abs(lon))
    
    gps_info = {
        1: 'N' if lat >= 0 else 'S',  # Latitude ref
        2: lat_dms,                    # Latitude
        3: 'E' if lon >= 0 else 'W',  # Longitude ref
        4: lon_dms,                    # Longitude
        5: 0,                          # Altitude ref (sea level)
        6: (0, 1),                     # Altitude
    }
    return gps_info


def add_exif_to_image(image_path, timestamp, lat, lon):
    """Add EXIF data to an image."""
    try:
        img = Image.open(image_path)
        
        # Convert to RGB if necessary (for JPEG compatibility)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Create EXIF dict if it doesn't exist
        exif = img.info.get('exif', b'')
        
        from PIL import ExifTags
        
        # Get or create exif data
        if hasattr(img, '_getexif') and img._getexif():
            exif_dict = {TAGS.get(k, k): v for k, v in img._getexif().items()}
        else:
            exif_dict = {}
        
        # Add timestamp
        exif_dict['DateTimeOriginal'] = timestamp
        exif_dict['DateTime'] = timestamp
        exif_dict['DateTimeDigitized'] = timestamp
        
        # Create new EXIF bytes
        # We'll use piexif library for better EXIF handling
        try:
            import piexif
            
            # Build GPS info
            gps_ifd = create_gps_exif(lat, lon)
            
            exif_dict_new = {
                "0th": {
                    piexif.ImageIFD.DateTime: timestamp.encode(),
                    piexif.ImageIFD.Make: "FakeCamera".encode(),
                    piexif.ImageIFD.Model: "ModelX".encode(),
                },
                "Exif": {
                    piexif.ExifIFD.DateTimeOriginal: timestamp.encode(),
                    piexif.ExifIFD.DateTimeDigitized: timestamp.encode(),
                },
                "GPS": gps_ifd,
                "1st": {},
                "thumbnail": None,
            }
            
            exif_bytes = piexif.dump(exif_dict_new)
            piexif.insert(exif_bytes, image_path)
            
        except ImportError:
            # Fallback without piexif
            print(f"  piexif not available, using PIL only for {image_path}")
            # Save with basic EXIF
            img.save(image_path, 'JPEG', exif=exif)
        
        return True
    except Exception as e:
        print(f"Error adding EXIF to {image_path}: {e}")
        return False


def generate_random_timestamp():
    """Generate a random timestamp within the last 5 years."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    random_days = random.randint(0, (end_date - start_date).days)
    random_time = timedelta(
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
        seconds=random.randint(0, 59)
    )
    
    random_date = start_date + timedelta(days=random_days) + random_time
    return random_date.strftime("%Y:%m:%d %H:%M:%S")


def main():
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png'}
    image_files = [f for f in os.listdir(DATASET_DIR) 
                   if os.path.splitext(f.lower())[1] in image_extensions]
    
    print(f"Found {len(image_files)} images\n")
    print("-" * 80)
    
    # Track status
    has_exif = []
    needs_exif = []
    
    for img_file in sorted(image_files):
        img_path = os.path.join(DATASET_DIR, img_file)
        timestamp, gps_info = get_existing_exif(img_path)
        
        if timestamp and gps_info:
            has_exif.append((img_file, timestamp))
            print(f"✓ {img_file}: Already has EXIF")
            print(f"    Timestamp: {timestamp}")
        elif timestamp:
            has_exif.append((img_file, timestamp))
            print(f"⚠ {img_file}: Has timestamp but no GPS")
            print(f"    Timestamp: {timestamp}")
            needs_exif.append(img_file)
        elif gps_info:
            print(f"⚠ {img_file}: Has GPS but no timestamp")
            needs_exif.append(img_file)
        else:
            print(f"✗ {img_file}: No EXIF data")
            needs_exif.append(img_file)
    
    print("\n" + "=" * 80)
    print(f"\nImages with EXIF: {len(has_exif)}")
    print(f"Images needing EXIF: {len(needs_exif)}")
    
    if not needs_exif:
        print("\nAll images already have EXIF data!")
        return
    
    print("\n" + "=" * 80)
    print("Adding EXIF data to images without it...\n")
    
    random_location_idx = 0
    
    for img_file in needs_exif:
        img_path = os.path.join(DATASET_DIR, img_file)
        
        # Determine location
        if img_file in LANDMARKS:
            lat, lon = LANDMARKS[img_file]
            location_type = "Landmark"
        else:
            # Use random location, cycle through list
            lat, lon = RANDOM_LOCATIONS[random_location_idx % len(RANDOM_LOCATIONS)]
            random_location_idx += 1
            location_type = "Random"
        
        # Generate random timestamp
        timestamp = generate_random_timestamp()
        
        print(f"Adding to {img_file}:")
        print(f"  Location: {location_type} ({lat:.4f}, {lon:.4f})")
        print(f"  Timestamp: {timestamp}")
        
        success = add_exif_to_image(img_path, timestamp, lat, lon)
        if success:
            print(f"  ✓ Success\n")
        else:
            print(f"  ✗ Failed\n")
    
    print("Done!")


if __name__ == "__main__":
    main()
