"""City and distance data for CRA Scheduler Environment.

Uses real US city coordinates to compute driving distances and travel times.
Supports 500+ sites across the US.
"""

import math
from typing import Tuple

# Real US cities with (lat, lon) — covers major metro areas and clinical trial hubs
# Format: "City, State": (latitude, longitude)
CITY_COORDS = {
    # Northeast
    "Trenton, NJ": (40.2171, -74.7429),
    "Philadelphia, PA": (39.9526, -75.1652),
    "NYC, NY": (40.7128, -74.0060),
    "Boston, MA": (42.3601, -71.0589),
    "Baltimore, MD": (39.2904, -76.6122),
    "Washington, DC": (38.9072, -77.0369),
    "Hartford, CT": (41.7658, -72.6734),
    "Pittsburgh, PA": (40.4406, -79.9959),
    "Richmond, VA": (37.5407, -77.4360),
    "Albany, NY": (42.6526, -73.7562),
    "Newark, NJ": (40.7357, -74.1724),
    "Providence, RI": (41.8240, -71.4128),
    "Buffalo, NY": (42.8864, -78.8784),
    "Syracuse, NY": (43.0481, -76.1474),
    "Wilmington, DE": (39.7391, -75.5398),
    "Atlantic City, NJ": (39.3643, -74.4229),
    "Portland, ME": (43.6591, -70.2568),
    "Burlington, VT": (44.4759, -73.2121),
    "New Haven, CT": (41.3083, -72.9279),
    "Scranton, PA": (41.4090, -75.6624),
    # Southeast
    "Charlotte, NC": (35.2271, -80.8431),
    "Raleigh, NC": (35.7796, -78.6382),
    "Atlanta, GA": (33.7490, -84.3880),
    "Miami, FL": (25.7617, -80.1918),
    "Tampa, FL": (27.9506, -82.4572),
    "Orlando, FL": (28.5383, -81.3792),
    "Jacksonville, FL": (30.3322, -81.6557),
    "Nashville, TN": (36.1627, -86.7816),
    "Charleston, SC": (32.7765, -79.9311),
    "Savannah, GA": (32.0809, -81.0912),
    "Norfolk, VA": (36.8508, -76.2859),
    "Knoxville, TN": (35.9606, -83.9207),
    "Memphis, TN": (35.1495, -90.0490),
    "Birmingham, AL": (33.5207, -86.8025),
    "New Orleans, LA": (29.9511, -90.0715),
    "Louisville, KY": (38.2527, -85.7585),
    "Lexington, KY": (38.0406, -84.5037),
    "Columbia, SC": (34.0007, -81.0348),
    "Greenville, SC": (34.8526, -82.3940),
    "Chattanooga, TN": (35.0456, -85.3097),
    # Midwest
    "Chicago, IL": (41.8781, -87.6298),
    "Detroit, MI": (42.3314, -83.0458),
    "Cleveland, OH": (41.4993, -81.6944),
    "Columbus, OH": (39.9612, -82.9988),
    "Cincinnati, OH": (39.1031, -84.5120),
    "Indianapolis, IN": (39.7684, -86.1581),
    "Milwaukee, WI": (43.0389, -87.9065),
    "Minneapolis, MN": (44.9778, -93.2650),
    "St. Louis, MO": (38.6270, -90.1994),
    "Kansas City, MO": (39.0997, -94.5786),
    "Omaha, NE": (41.2565, -95.9345),
    "Des Moines, IA": (41.5868, -93.6250),
    "Madison, WI": (43.0731, -89.4012),
    "Grand Rapids, MI": (42.9634, -85.6681),
    "Ann Arbor, MI": (42.2808, -83.7430),
    "Dayton, OH": (39.7589, -84.1916),
    "Toledo, OH": (41.6528, -83.5379),
    "Fort Wayne, IN": (41.0793, -85.1394),
    "Springfield, IL": (39.7817, -89.6501),
    "Peoria, IL": (40.6936, -89.5890),
    # Southwest / West
    "Dallas, TX": (32.7767, -96.7970),
    "Houston, TX": (29.7604, -95.3698),
    "San Antonio, TX": (29.4241, -98.4936),
    "Austin, TX": (30.2672, -97.7431),
    "Phoenix, AZ": (33.4484, -112.0740),
    "Denver, CO": (39.7392, -104.9903),
    "Salt Lake City, UT": (40.7608, -111.8910),
    "Albuquerque, NM": (35.0844, -106.6504),
    "Tucson, AZ": (32.2226, -110.9747),
    "El Paso, TX": (31.7619, -106.4850),
    "Oklahoma City, OK": (35.4676, -97.5164),
    "Tulsa, OK": (36.1540, -95.9928),
    "Colorado Springs, CO": (38.8339, -104.8214),
    "Wichita, KS": (37.6872, -97.3301),
    "Little Rock, AR": (34.7465, -92.2896),
    "Boise, ID": (43.6150, -116.2023),
    "Lubbock, TX": (33.5779, -101.8552),
    "Amarillo, TX": (35.2220, -101.8313),
    "Fort Worth, TX": (32.7555, -97.3308),
    "Corpus Christi, TX": (27.8006, -97.3964),
    # West Coast
    "Los Angeles, CA": (34.0522, -118.2437),
    "San Francisco, CA": (37.7749, -122.4194),
    "San Diego, CA": (32.7157, -117.1611),
    "Seattle, WA": (47.6062, -122.3321),
    "Portland, OR": (45.5152, -122.6784),
    "Sacramento, CA": (38.5816, -121.4944),
    "Las Vegas, NV": (36.1699, -115.1398),
    "San Jose, CA": (37.3382, -121.8863),
    "Fresno, CA": (36.7378, -119.7871),
    "Long Beach, CA": (33.7701, -118.1937),
    "Oakland, CA": (37.8044, -122.2712),
    "Riverside, CA": (33.9533, -117.3962),
    "Bakersfield, CA": (35.3733, -119.0187),
    "Spokane, WA": (47.6588, -117.4260),
    "Tacoma, WA": (47.2529, -122.4443),
    "Reno, NV": (39.5296, -119.8138),
    "Eugene, OR": (44.0521, -123.0868),
    "Santa Barbara, CA": (34.4208, -119.6982),
    "Irvine, CA": (33.6846, -117.8265),
    "Pasadena, CA": (34.1478, -118.1445),
}

ALL_CITIES = list(CITY_COORDS.keys())


def _haversine(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """Haversine distance in miles between two (lat, lon) points."""
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 3959 * 2 * math.asin(math.sqrt(a))


def get_distance(city_a: str, city_b: str) -> int:
    """Get driving distance estimate between two cities (miles).

    Uses haversine * 1.3 factor to approximate driving distance.
    """
    if city_a == city_b:
        return 0
    straight = _haversine(CITY_COORDS[city_a], CITY_COORDS[city_b])
    return round(straight * 1.3)  # driving factor


def get_travel_time(city_a: str, city_b: str) -> float:
    """Get estimated travel time in hours (assumes avg 50 mph driving)."""
    return round(get_distance(city_a, city_b) / 50, 1)


def get_travel_days(city_a: str, city_b: str) -> int:
    """Get travel time in days (8 hours driving per day, minimum 1 day)."""
    hours = get_travel_time(city_a, city_b)
    if hours == 0:
        return 0
    return max(1, math.ceil(hours / 8))


# Max distance in the dataset (for reward normalization)
MAX_DISTANCE = max(
    get_distance(a, b)
    for a in list(CITY_COORDS.keys())[:20]
    for b in list(CITY_COORDS.keys())[:20]
    if a != b
)
