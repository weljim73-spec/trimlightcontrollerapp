"""
Trimlight Edge API Client
Handles authentication and API communication with Trimlight devices.
"""

import hmac
import hashlib
import base64
import time
import requests
from typing import Optional, Dict, Any, List
import streamlit as st


class TrimlightAPI:
    """Client for interacting with the Trimlight Edge API."""

    BASE_URL = "https://trimlight.ledhue.com/trimlight"

    def __init__(self, client_id: str, client_secret: str):
        """Initialize the API client with credentials."""
        self.client_id = client_id
        self.client_secret = client_secret
        self._session = requests.Session()

    def _generate_auth_token(self, timestamp: int) -> str:
        """Generate the HMAC-SHA256 authentication token."""
        message = f"Trimlight|{self.client_id}|{timestamp}"
        signature = hmac.new(
            self.client_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).digest()
        return base64.b64encode(signature).decode('utf-8')

    def _get_headers(self) -> Dict[str, str]:
        """Generate request headers with authentication."""
        timestamp = int(time.time() * 1000)
        token = self._generate_auth_token(timestamp)
        return {
            "authorization": token,
            "S-ClientId": self.client_id,
            "S-Timestamp": str(timestamp),
            "Content-Type": "application/json"
        }

    def _request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        """Make an authenticated request to the API."""
        url = f"{self.BASE_URL}{endpoint}"
        headers = self._get_headers()

        try:
            if method == "GET":
                response = self._session.get(url, headers=headers, params=data, timeout=30)
            else:
                response = self._session.post(url, headers=headers, json=data, timeout=30)

            response.raise_for_status()
            result = response.json()

            if result.get("code") != 0:
                raise APIError(result.get("desc", "Unknown error"), result.get("code"))

            return result
        except requests.exceptions.RequestException as e:
            raise APIError(f"Request failed: {str(e)}")

    # Device Methods
    def get_devices(self, page: int = 0) -> Dict[str, Any]:
        """Get list of all devices."""
        return self._request("GET", "/v1/oauth/resources/devices", {"page": page})

    def get_device_details(self, device_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific device."""
        return self._request("POST", "/v1/oauth/resources/device/get", {
            "deviceId": device_id,
            "currentDate": self._get_current_date()
        })

    def set_device_switch_state(self, device_id: str, state: int) -> Dict[str, Any]:
        """Set device switch state (0: off, 1: manual, 2: timer)."""
        return self._request("POST", "/v1/oauth/resources/device/update", {
            "deviceId": device_id,
            "payload": {"switchState": state}
        })

    def set_device_name(self, device_id: str, name: str) -> Dict[str, Any]:
        """Set device name."""
        return self._request("POST", "/v1/oauth/resources/device/update", {
            "deviceId": device_id,
            "payload": {"name": name}
        })

    # Effect Methods
    def preview_builtin_effect(self, device_id: str, mode: int, speed: int = 100,
                                brightness: int = 100, pixel_len: int = 30,
                                reverse: bool = False) -> Dict[str, Any]:
        """Preview a built-in effect on the device."""
        return self._request("POST", "/v1/oauth/resources/device/effect/preview", {
            "deviceId": device_id,
            "payload": {
                "category": 0,
                "mode": mode,
                "speed": speed,
                "brightness": brightness,
                "pixelLen": pixel_len,
                "reverse": reverse
            }
        })

    def preview_custom_effect(self, device_id: str, mode: int, speed: int = 100,
                              brightness: int = 100, pixels: List[Dict] = None) -> Dict[str, Any]:
        """Preview a custom effect on the device."""
        return self._request("POST", "/v1/oauth/resources/device/effect/preview", {
            "deviceId": device_id,
            "payload": {
                "category": 1,
                "mode": mode,
                "speed": speed,
                "brightness": brightness,
                "pixels": pixels or []
            }
        })

    def view_effect(self, device_id: str, effect_id: int) -> Dict[str, Any]:
        """Activate/view a saved effect on the device."""
        return self._request("POST", "/v1/oauth/resources/device/effect/view", {
            "deviceId": device_id,
            "payload": {"id": effect_id}
        })

    def save_effect(self, device_id: str, effect_data: Dict) -> Dict[str, Any]:
        """Save an effect to the device."""
        return self._request("POST", "/v1/oauth/resources/device/effect/save", {
            "deviceId": device_id,
            "payload": effect_data
        })

    def delete_effect(self, device_id: str, effect_id: int) -> Dict[str, Any]:
        """Delete an effect from the device."""
        return self._request("POST", "/v1/oauth/resources/device/effect/delete", {
            "deviceId": device_id,
            "payload": {"id": effect_id}
        })

    # Schedule Methods
    def update_daily_schedule(self, device_id: str, schedule_data: Dict) -> Dict[str, Any]:
        """Update a daily schedule."""
        return self._request("POST", "/v1/oauth/resources/device/daily/save", {
            "deviceId": device_id,
            "payload": schedule_data
        })

    def save_calendar_schedule(self, device_id: str, schedule_data: Dict) -> Dict[str, Any]:
        """Save a calendar schedule."""
        return self._request("POST", "/v1/oauth/resources/device/calendar/save", {
            "deviceId": device_id,
            "payload": schedule_data
        })

    def delete_calendar_schedule(self, device_id: str, schedule_id: int) -> Dict[str, Any]:
        """Delete a calendar schedule."""
        return self._request("POST", "/v1/oauth/resources/device/calendar/delete", {
            "deviceId": device_id,
            "payload": {"id": schedule_id}
        })

    # Utility Methods
    def _get_current_date(self) -> Dict[str, int]:
        """Get current date in API format."""
        now = time.localtime()
        weekday_map = {0: 2, 1: 3, 2: 4, 3: 5, 4: 6, 5: 7, 6: 1}  # Python to API weekday
        return {
            "year": now.tm_year - 2000,
            "month": now.tm_mon,
            "day": now.tm_mday,
            "weekday": weekday_map[now.tm_wday],
            "hours": now.tm_hour,
            "minutes": now.tm_min,
            "seconds": now.tm_sec
        }

    def notify_update_shadow(self, device_id: str) -> Dict[str, Any]:
        """Notify device to update shadow data."""
        return self._request("GET", "/v1/oauth/resources/device/notify-update-shadow", {
            "deviceId": device_id,
            "currentDate": self._get_current_date()
        })


class APIError(Exception):
    """Custom exception for API errors."""
    def __init__(self, message: str, code: int = None):
        self.message = message
        self.code = code
        super().__init__(self.message)


# Built-in effect definitions
BUILTIN_EFFECTS = {
    0: "Rainbow Gradual Chase", 1: "Rainbow Comet", 2: "Rainbow Segment",
    3: "Rainbow Wave", 4: "Rainbow Meteor", 5: "Rainbow Gradual",
    6: "Rainbow Jump", 7: "Rainbow Stars", 8: "Rainbow Fade In Out",
    9: "Rainbow Spin", 10: "Red Stacking", 11: "Green Stacking",
    12: "Blue Stacking", 13: "Yellow Stacking", 14: "Cyan Stacking",
    15: "Purple Stacking", 16: "White Stacking", 17: "Full Color Stack",
    18: "Red to Green Stack", 19: "Green to Blue Stack", 20: "Blue to Yellow Stack",
    21: "Yellow to Cyan Stack", 22: "Cyan to Purple Stack", 23: "Purple to White Stack",
    24: "Red Comet", 25: "Green Comet", 26: "Blue Comet",
    27: "Yellow Comet", 28: "Cyan Comet", 29: "Purple Comet",
    30: "White Comet", 31: "Red Meteor", 32: "Green Meteor",
    33: "Blue Meteor", 34: "Yellow Meteor", 35: "Cyan Meteor",
    36: "Purple Meteor", 37: "White Meteor", 38: "Red Wave",
    39: "Green Wave", 40: "Blue Wave", 41: "Yellow Wave",
    42: "Cyan Wave", 43: "Purple Wave", 44: "White Wave",
    45: "Red Green Wave", 46: "Red Blue Wave", 47: "Red Yellow Wave",
    48: "Red Cyan Wave", 49: "Red Purple Wave", 50: "Red White Wave",
    51: "Green Blue Wave", 52: "Green Yellow Wave", 53: "Green Cyan Wave",
    54: "Green Purple Wave", 55: "Green White Wave", 56: "Blue Yellow Wave",
    57: "Blue Cyan Wave", 58: "Blue Purple Wave", 59: "Blue White Wave",
    60: "Yellow Cyan Wave", 61: "Yellow Purple Wave", 62: "Yellow White Wave",
    63: "Cyan Purple Wave", 64: "Cyan White Wave", 65: "Purple White Wave",
    66: "Red Dot Pulse", 67: "Green Dot Pulse", 68: "Blue Dot Pulse",
    69: "Yellow Dot Pulse", 70: "Cyan Dot Pulse", 71: "Purple Dot Pulse",
    72: "White Dot Pulse", 73: "Red Green Blank Pulse", 74: "Green Blue Blank Pulse",
    75: "Blue Yellow Blank Pulse", 76: "Yellow Cyan Blank Pulse", 77: "Cyan Purple Blank Pulse",
    78: "Purple White Blank Pulse", 79: "Red with Purple Pulse", 80: "Green with Cyan Pulse",
    81: "Blue with Yellow Pulse", 82: "Yellow with Blue Pulse", 83: "Cyan with Green Pulse",
    84: "Purple with Purple Pulse", 85: "Red Comet Spin", 86: "Green Comet Spin",
    87: "Blue Comet Spin", 88: "Yellow Comet Spin", 89: "Cyan Comet Spin",
    90: "Purple Comet Spin", 91: "White Comet Spin", 92: "Red Dot Spin",
    93: "Green Dot Spin", 94: "Blue Dot Spin", 95: "Yellow Dot Spin",
    96: "Cyan Dot Spin", 97: "Purple Dot Spin", 98: "White Dot Spin",
    99: "Red Segment Spin", 100: "Green Segment Spin", 101: "Blue Segment Spin",
    102: "Yellow Segment Spin", 103: "Cyan Segment Spin", 104: "Purple Segment Spin",
    105: "White Segment Spin", 106: "Red Green Gradual Snake", 107: "Red Blue Gradual Snake",
    108: "Red Yellow Gradual Snake", 109: "Red Cyan Gradual Snake", 110: "Red Purple Gradual Snake",
    111: "Red White Gradual Snake", 112: "Green Blue Gradual Snake", 113: "Green Yellow Gradual Snake",
    114: "Green Cyan Gradual Snake", 115: "Green Purple Gradual Snake", 116: "Green White Gradual Snake",
    117: "Blue Yellow Gradual Snake", 118: "Blue Cyan Gradual Snake", 119: "Blue Purple Gradual Snake",
    120: "Blue White Gradual Snake", 121: "Yellow Cyan Gradual Snake", 122: "Yellow Purple Gradual Snake",
    123: "Yellow White Gradual Snake", 124: "Cyan Purple Gradual Snake", 125: "Cyan White Gradual Snake",
    126: "Purple White Gradual Snake", 127: "Red White Blank Snake", 128: "Green White Blank Snake",
    129: "Blue White Blank Snake", 130: "Yellow White Blank Snake", 131: "Cyan White Blank Snake",
    132: "Purple White Blank Snake", 133: "Green Yellow White Snake", 134: "Red Green White Snake",
    135: "Red Yellow Snake", 136: "Red White Snake", 137: "Green White Snake",
    138: "Red Stars", 139: "Green Stars", 140: "Blue Stars",
    141: "Yellow Stars", 142: "Cyan Stars", 143: "Purple Stars",
    144: "White Stars", 145: "Red Background Stars", 146: "Green Background Stars",
    147: "Blue Background Stars", 148: "Yellow Background Stars", 149: "Cyan Background Stars",
    150: "Purple Background Stars", 151: "Red White Background Stars", 152: "Green White Background Stars",
    153: "Blue White Background Stars", 154: "Yellow White Background Stars", 155: "Cyan White Background Stars",
    156: "Purple White Background Stars", 157: "White White Background Stars", 158: "Red Breath",
    159: "Green Breath", 160: "Blue Breath", 161: "Yellow Breath",
    162: "Cyan Breath", 163: "Purple Breath", 164: "White Breath",
    165: "Red Yellow Fire", 166: "Red Purple Fire", 167: "Green Yellow Fire",
    168: "Green Cyan Fire", 169: "Blue Purple Fire", 170: "Blue Cyan Fire",
    171: "Red Strobe", 172: "Green Strobe", 173: "Blue Strobe",
    174: "Yellow Strobe", 175: "Cyan Strobe", 176: "Purple Strobe",
    177: "White Strobe", 178: "Red Blue White Strobe", 179: "Full Color Strobe"
}

# Custom effect modes
CUSTOM_EFFECT_MODES = {
    0: "Static", 1: "Chase Forward", 2: "Chase Backward",
    3: "Chase Middle to Out", 4: "Chase Out to Middle", 5: "Stars",
    6: "Breath", 7: "Comet Forward", 8: "Comet Backward",
    9: "Comet Middle to Out", 10: "Comet Out to Middle", 11: "Wave Forward",
    12: "Wave Backward", 13: "Wave Middle to Out", 14: "Wave Out to Middle",
    15: "Strobe", 16: "Solid Fade"
}

# Effect categories for grouping
EFFECT_CATEGORIES = {
    "Rainbow": list(range(0, 10)),
    "Stacking": list(range(10, 24)),
    "Comet": list(range(24, 31)),
    "Meteor": list(range(31, 38)),
    "Wave": list(range(38, 66)),
    "Pulse": list(range(66, 85)),
    "Spin": list(range(85, 106)),
    "Snake": list(range(106, 138)),
    "Stars": list(range(138, 158)),
    "Breath": list(range(158, 165)),
    "Fire": list(range(165, 171)),
    "Strobe": list(range(171, 180))
}


def get_api_client() -> Optional[TrimlightAPI]:
    """Get API client from Streamlit secrets or session state."""
    try:
        if "trimlight" in st.secrets:
            client_id = st.secrets["trimlight"]["client_id"]
            client_secret = st.secrets["trimlight"]["client_secret"]
        elif "TRIMLIGHT_CLIENT_ID" in st.secrets:
            client_id = st.secrets["TRIMLIGHT_CLIENT_ID"]
            client_secret = st.secrets["TRIMLIGHT_CLIENT_SECRET"]
        else:
            return None

        return TrimlightAPI(client_id, client_secret)
    except Exception:
        return None
