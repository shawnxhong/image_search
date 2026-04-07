#!/usr/bin/env python3
"""
Add synthetic EXIF data (DateTimeOriginal and GPS info) to JPG files.
Only includes DateTimeOriginal and GPS tags, no other EXIF fields.
"""

import os
import random
from datetime import datetime, timedelta
from PIL import Image
from PIL.TiffImagePlugin import IFDRational

SOURCE_DIR = r"C:\Users\53422\Downloads\lfw_jpgs_random100"

# GPS range for random coordinates (US continental range as example)
LAT_MIN, LAT_MAX = 25.0, 49.0  # Latitude: ~Hawaii to Maine
LON_MIN, LON_MAX = -125.0, -66.0  # Longitude: ~West coast to East coast


def generate_random_datetime():
    """Generate a random datetime between 2015 and 2024."""
    start_date = datetime(2015, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    days_between = (end_date - start_date).days
    random_days = random.randint(0, days_between)
    random_time = timedelta(
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
        seconds=random.randint(0, 59)
    )
    
    result = start_date + timedelta(days=random_days) + random_time
    return result.strftime("%Y:%m:%d %H:%M:%S")


def decimal_to_dms(decimal_coord):
    """Convert decimal coordinate to degrees, minutes, seconds tuple."""
    degrees = int(abs(decimal_coord))
    minutes_float = (abs(decimal_coord) - degrees) * 60
    minutes = int(minutes_float)
    seconds = round((minutes_float - minutes) * 60, 3)
    
    # Return as tuples of IFDRational (numerator, denominator)
    return (
        IFDRational(degrees, 1),
        IFDRational(minutes, 1),
        IFDRational(int(seconds * 1000), 1000)
    )


def generate_random_gps():
    """Generate random GPS coordinates."""
    lat = random.uniform(LAT_MIN, LAT_MAX)
    lon = random.uniform(LON_MIN, LON_MAX)
    
    lat_ref = 'N' if lat >= 0 else 'S'
    lon_ref = 'E' if lon >= 0 else 'W'
    
    return {
        'lat': decimal_to_dms(lat),
        'lat_ref': lat_ref,
        'lon': decimal_to_dms(lon),
        'lon_ref': lon_ref
    }


def add_exif_to_image(image_path):
    """Add synthetic EXIF data to a single image."""
    try:
        img = Image.open(image_path)
        
        # Generate synthetic data
        datetime_str = generate_random_datetime()
        gps_data = generate_random_gps()
        
        # Create EXIF using PIL.Image.Exif
        exif = Image.Exif()
        
        # DateTimeOriginal tag = 36867 (0x9003) - "Date taken" in Windows
        exif[36867] = datetime_str
        
        # DateTime tag = 306 (0x0132) - helps Windows display "Date taken"
        exif[306] = datetime_str
        
        # DateTimeDigitized tag = 36868 (0x9004) - also helps with "Date taken"
        exif[36868] = datetime_str
        
        # GPSInfo - nested dictionary
        # GPS tags: 1=lat_ref, 2=lat, 3=lon_ref, 4=lon
        exif[34853] = {
            1: gps_data['lat_ref'],
            2: gps_data['lat'],
            3: gps_data['lon_ref'],
            4: gps_data['lon'],
        }
        
        # Save image with EXIF
        img.save(image_path, exif=exif)
        img.close()
        
        return True, datetime_str, gps_data
    except Exception as e:
        return False, str(e), None


def main():
    # Get all JPG files
    jpg_files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith('.jpg')]
    jpg_files.sort()
    
    print(f"Found {len(jpg_files)} JPG files in {SOURCE_DIR}")
    print("-" * 60)
    
    success_count = 0
    
    for i, filename in enumerate(jpg_files, 1):
        filepath = os.path.join(SOURCE_DIR, filename)
        success, dt, gps = add_exif_to_image(filepath)
        
        if success:
            success_count += 1
            lat_dms = f"{gps['lat'][0]}°{gps['lat'][1]}'{gps['lat'][2]}\"{gps['lat_ref']}"
            lon_dms = f"{gps['lon'][0]}°{gps['lon'][1]}'{gps['lon'][2]}\"{gps['lon_ref']}"
            print(f"[{i:3d}/{len(jpg_files)}] {filename}")
            print(f"        DateTimeOriginal: {dt}")
            print(f"        GPS: {lat_dms}, {lon_dms}")
        else:
            print(f"[{i:3d}/{len(jpg_files)}] FAILED: {filename} - {dt}")
    
    print("-" * 60)
    print(f"Done! Successfully added EXIF data to {success_count}/{len(jpg_files)} images.")


if __name__ == "__main__":
    main()
