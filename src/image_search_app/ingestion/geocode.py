"""Reverse geocoding: GPS coordinates to country/state/city."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GeoLocation:
    country: str | None = None
    state: str | None = None
    city: str | None = None


def reverse_geocode(lat: float, lon: float) -> GeoLocation:
    """Convert GPS coordinates to country/state/city using geopy Nominatim.

    Returns a GeoLocation with whatever fields could be resolved.
    Falls back to empty fields on error — never raises.
    """
    try:
        from geopy.geocoders import Nominatim

        geolocator = Nominatim(user_agent="image_search_app", timeout=5)
        location = geolocator.reverse(f"{lat}, {lon}", language="en", exactly_one=True)
        if location is None:
            return GeoLocation()

        addr = location.raw.get("address", {})
        country = addr.get("country")
        state = addr.get("state")
        city = (
            addr.get("city")
            or addr.get("town")
            or addr.get("village")
            or addr.get("municipality")
        )
        return GeoLocation(country=country, state=state, city=city)

    except ImportError:
        logger.warning("geopy not installed — skipping reverse geocoding")
        return GeoLocation()
    except Exception as exc:
        logger.warning("Reverse geocoding failed for (%s, %s): %s", lat, lon, exc)
        return GeoLocation()
