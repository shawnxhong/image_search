#!/usr/bin/env python3
"""
Add synthetic EXIF data (DateTimeOriginal and GPS info) to JPG files.
Matches the structure of real camera photos for Windows compatibility.
"""

import os
import random
from datetime import datetime, timedelta
from PIL import Image
from PIL.TiffImagePlugin import IFDRational
from io import BytesIO

SOURCE_DIR = r"C:\Users\53422\Downloads\lfw_jpgs_random100"

# GPS range for random coordinates
LAT_MIN, LAT_MAX = 25.0, 49.0
LON_MIN, LON_MAX = -125.0, -66.0


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


def create_exif_bytes(datetime_str, gps_data):
    """
    Create EXIF bytes with proper IFD structure.
    This creates EXIF data similar to real camera photos.
    """
    from PIL.TiffImagePlugin import ImageFileDirectory_v2
    
    # Main IFD tags (IFD0)
    # Tag numbers: 306=DateTime, 34665=ExifOffset, 34853=GPSInfo
    ifd_main = ImageFileDirectory_v2()
    ifd_main[306] = datetime_str  # DateTime
    
    # GPSInfo IFD (tag 34853)
    ifd_gps = ImageFileDirectory_v2()
    ifd_gps[1] = gps_data['lat_ref']  # GPSLatitudeRef
    ifd_gps[2] = gps_data['lat']      # GPSLatitude
    ifd_gps[3] = gps_data['lon_ref']  # GPSLongitudeRef
    ifd_gps[4] = gps_data['lon']      # GPSLongitude
    
    # EXIF IFD (tag 34665) - contains DateTimeOriginal and DateTimeDigitized
    ifd_exif = ImageFileDirectory_v2()
    ifd_exif[36867] = datetime_str  # DateTimeOriginal
    ifd_exif[36868] = datetime_str  # DateTimeDigitized
    
    # Serialize the IFDs to bytes
    # We need to manually build the EXIF structure
    
    # First, serialize the EXIF IFD
    exif_data = {}
    exif_data[36867] = datetime_str
    exif_data[36868] = datetime_str
    
    # Serialize GPS IFD
    gps_data_dict = {
        1: gps_data['lat_ref'],
        2: gps_data['lat'],
        3: gps_data['lon_ref'],
        4: gps_data['lon'],
    }
    
    # Create the full EXIF bytes using TiffImagePlugin
    output = BytesIO()
    
    # Use little-endian byte order
    ifd_final = ImageFileDirectory_v2()
    ifd_final[306] = datetime_str
    ifd_final[34665] = exif_data  # ExifOffset pointing to sub-IFD
    ifd_final[34853] = gps_data_dict  # GPSInfo
    
    # Write the IFD to bytes
    ifd_final.write(output)
    
    return output.getvalue()


def add_exif_to_image(image_path):
    """Add synthetic EXIF data to a single image."""
    try:
        img = Image.open(image_path)
        
        # Generate synthetic data
        datetime_str = generate_random_datetime()
        gps_data = generate_random_gps()
        
        # Create EXIF with proper structure using dictionary
        exif = Image.Exif()
        
        # Main IFD tags
        exif[306] = datetime_str  # DateTime
        
        # EXIF sub-IFD (tag 34665) - this is crucial for Windows "Date taken"
        exif[34665] = {
            36867: datetime_str,  # DateTimeOriginal
            36868: datetime_str,  # DateTimeDigitized
        }
        
        # GPS IFD (tag 34853)
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
