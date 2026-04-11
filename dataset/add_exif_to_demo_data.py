#!/usr/bin/env python3
"""Add synthetic EXIF metadata to every image in demo_data.

- Obvious landmark photos get real GPS coordinates for that landmark.
- All other images get deterministic pseudo-random GPS coordinates.
- Timestamps are deterministic per file but otherwise arbitrary.

This script overwrites DateTime / DateTimeOriginal / DateTimeDigitized and GPS.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from PIL import Image
from PIL.TiffImagePlugin import IFDRational


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEMO_DATA_DIR = PROJECT_ROOT / "demo_data"


@dataclass(frozen=True)
class LocationSpec:
    label: str
    lat: float
    lon: float


LANDMARK_LOCATIONS: dict[str, LocationSpec] = {
    "2E8P1PA.jpg": LocationSpec("Eiffel Tower, Paris", 48.8584, 2.2945),
    "4423.jpg": LocationSpec("Eiffel Tower, Paris", 48.8584, 2.2945),
    "54339.jpg": LocationSpec("Eiffel Tower, Paris", 48.8584, 2.2945),
    "450.jpg": LocationSpec("Sydney Opera House, Sydney", -33.8568, 151.2153),
    "5101.jpg": LocationSpec("Sydney Opera House, Sydney", -33.8568, 151.2153),
    "154616.jpg": LocationSpec("Panmunjom, Korean DMZ", 37.9563, 126.6791),
    "154945.jpg": LocationSpec("Genghis Khan Equestrian Statue, Mongolia", 47.8081, 107.5367),
    "np_file_28275.jpeg": LocationSpec("Moscow Kremlin, Moscow", 55.7525, 37.6231),
}


RANDOM_LOCATIONS: list[LocationSpec] = [
    LocationSpec("Miami Beach, USA", 25.7907, -80.1300),
    LocationSpec("Santa Monica, USA", 34.0195, -118.4912),
    LocationSpec("Central Park, New York", 40.7812, -73.9665),
    LocationSpec("Kyoto, Japan", 35.0116, 135.7681),
    LocationSpec("Rome, Italy", 41.9028, 12.4964),
    LocationSpec("Singapore", 1.3521, 103.8198),
    LocationSpec("Seoul, South Korea", 37.5665, 126.9780),
    LocationSpec("Bangkok, Thailand", 13.7563, 100.5018),
    LocationSpec("Cape Town, South Africa", -33.9249, 18.4241),
    LocationSpec("Vancouver, Canada", 49.2827, -123.1207),
    LocationSpec("Berlin, Germany", 52.5200, 13.4050),
    LocationSpec("Rio de Janeiro, Brazil", -22.9068, -43.1729),
    LocationSpec("Auckland, New Zealand", -36.8509, 174.7645),
    LocationSpec("Barcelona, Spain", 41.3874, 2.1686),
    LocationSpec("Istanbul, Turkey", 41.0082, 28.9784),
]


def file_rng(name: str) -> random.Random:
    seed = int(hashlib.sha256(name.encode("utf-8")).hexdigest()[:16], 16)
    return random.Random(seed)


def pick_location(name: str) -> LocationSpec:
    landmark = LANDMARK_LOCATIONS.get(name)
    if landmark is not None:
        return landmark

    rng = file_rng(f"gps::{name}")
    return RANDOM_LOCATIONS[rng.randrange(len(RANDOM_LOCATIONS))]


def generate_timestamp(name: str) -> str:
    rng = file_rng(f"time::{name}")
    start = datetime(2018, 1, 1, 0, 0, 0)
    end = datetime(2025, 12, 31, 23, 59, 59)
    seconds = int((end - start).total_seconds())
    dt = start + timedelta(seconds=rng.randint(0, seconds))
    return dt.strftime("%Y:%m:%d %H:%M:%S")


def decimal_to_dms(decimal_coord: float) -> tuple[IFDRational, IFDRational, IFDRational]:
    absolute = abs(decimal_coord)
    degrees = int(absolute)
    minutes_float = (absolute - degrees) * 60
    minutes = int(minutes_float)
    seconds = round((minutes_float - minutes) * 60, 4)
    return (
        IFDRational(degrees, 1),
        IFDRational(minutes, 1),
        IFDRational(int(seconds * 10000), 10000),
    )


def write_exif(image_path: Path, timestamp: str, location: LocationSpec) -> None:
    with Image.open(image_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")

        exif = img.getexif()
        exif[306] = timestamp
        exif[36867] = timestamp
        exif[36868] = timestamp
        exif[34853] = {
            1: "N" if location.lat >= 0 else "S",
            2: decimal_to_dms(location.lat),
            3: "E" if location.lon >= 0 else "W",
            4: decimal_to_dms(location.lon),
        }
        img.save(image_path, exif=exif)


def iter_demo_images() -> list[Path]:
    return sorted(
        p for p in DEMO_DATA_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}
    )


def main() -> None:
    images = iter_demo_images()
    print(f"Found {len(images)} demo images in {DEMO_DATA_DIR}")
    print("-" * 72)

    for idx, image_path in enumerate(images, start=1):
        timestamp = generate_timestamp(image_path.name)
        location = pick_location(image_path.name)
        write_exif(image_path, timestamp, location)
        gps_type = "landmark" if image_path.name in LANDMARK_LOCATIONS else "random"
        print(
            f"[{idx:02d}/{len(images):02d}] {image_path.name}\n"
            f"  time: {timestamp}\n"
            f"  gps : {location.lat:.4f}, {location.lon:.4f} ({gps_type}: {location.label})"
        )

    print("-" * 72)
    print("EXIF update complete.")


if __name__ == "__main__":
    main()
