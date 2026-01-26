"""
Trimlight LED Configuration App
A Streamlit app for configuring and visualizing Trimlight home exterior LED lights.
"""

import streamlit as st
import time
from typing import Optional, Dict, List, Any

from trimlight_api import (
    TrimlightAPI, APIError, get_api_client,
    BUILTIN_EFFECTS, CUSTOM_EFFECT_MODES, EFFECT_CATEGORIES
)
from led_visualizer import (
    EffectRenderer, generate_linear_strip_svg, generate_house_outline_svg,
    int_to_rgb, rgb_to_hex, hex_to_rgb, rgb_to_int, apply_brightness
)

# Page configuration
st.set_page_config(
    page_title="Trimlight Controller",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .device-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid #2a2a4a;
    }
    .status-online {
        color: #00ff88;
        font-weight: bold;
    }
    .status-offline {
        color: #ff4444;
        font-weight: bold;
    }
    .effect-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
        gap: 10px;
    }
    .preview-container {
        background: #1a1a2e;
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
    }
    div[data-testid="stHorizontalBlock"] {
        align-items: center;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'api_client': None,
        'devices': [],
        'selected_device': None,
        'device_details': None,
        'preview_frame': 0,
        'is_animating': False,
        'num_leds': 150,
        'view_type': 'linear',
        'custom_pixels': [],
        'last_refresh': 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_or_create_api_client() -> Optional[TrimlightAPI]:
    """Get API client from session state or create new one."""
    if st.session_state.api_client is None:
        st.session_state.api_client = get_api_client()
    return st.session_state.api_client


def refresh_devices(api: TrimlightAPI):
    """Refresh the device list from API."""
    try:
        result = api.get_devices()
        st.session_state.devices = result.get('payload', {}).get('data', [])
        st.session_state.last_refresh = time.time()
    except APIError as e:
        st.error(f"Failed to fetch devices: {e.message}")


def refresh_device_details(api: TrimlightAPI, device_id: str):
    """Refresh details for a specific device."""
    try:
        result = api.get_device_details(device_id)
        st.session_state.device_details = result.get('payload', {})
    except APIError as e:
        st.error(f"Failed to fetch device details: {e.message}")


def render_device_selector(api: TrimlightAPI):
    """Render device selection sidebar."""
    st.sidebar.header("🏠 Devices")

    col1, col2 = st.sidebar.columns([3, 1])
    with col2:
        if st.button("🔄", help="Refresh devices", key="refresh_devices_btn"):
            refresh_devices(api)

    if not st.session_state.devices:
        refresh_devices(api)

    devices = st.session_state.devices

    if not devices:
        st.sidebar.warning("No devices found")
        return

    # Device selection
    device_options = {d['deviceId']: d['name'] for d in devices}
    selected_id = st.sidebar.selectbox(
        "Select Device",
        options=list(device_options.keys()),
        format_func=lambda x: device_options[x],
        key="device_selector"
    )

    if selected_id != st.session_state.selected_device:
        st.session_state.selected_device = selected_id
        refresh_device_details(api, selected_id)

    # Device info
    if st.session_state.selected_device:
        device = next((d for d in devices if d['deviceId'] == selected_id), None)
        if device:
            st.sidebar.markdown("---")
            status = "🟢 Online" if device.get('connectivity') == 1 else "🔴 Offline"
            st.sidebar.markdown(f"**Status:** {status}")

            switch_state = device.get('switchState', 0)
            state_text = {0: "Off", 1: "Manual", 2: "Timer"}.get(switch_state, "Unknown")
            st.sidebar.markdown(f"**Mode:** {state_text}")

            st.sidebar.markdown(f"**Firmware:** {device.get('fwVersionName', 'N/A')}")


def render_device_controls(api: TrimlightAPI):
    """Render device control panel."""
    if not st.session_state.device_details:
        return

    details = st.session_state.device_details
    device_id = st.session_state.selected_device

    st.subheader("⚡ Device Controls")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Power controls
        st.markdown("**Power Mode**")
        current_state = details.get('switchState', 0)

        power_col1, power_col2, power_col3 = st.columns(3)

        with power_col1:
            if st.button("Off", use_container_width=True,
                        type="primary" if current_state == 0 else "secondary",
                        key="power_off_btn"):
                try:
                    api.set_device_switch_state(device_id, 0)
                    st.success("Lights turned off")
                    refresh_device_details(api, device_id)
                except APIError as e:
                    st.error(f"Error: {e.message}")

        with power_col2:
            if st.button("Manual", use_container_width=True,
                        type="primary" if current_state == 1 else "secondary",
                        key="power_manual_btn"):
                try:
                    api.set_device_switch_state(device_id, 1)
                    st.success("Manual mode enabled")
                    refresh_device_details(api, device_id)
                except APIError as e:
                    st.error(f"Error: {e.message}")

        with power_col3:
            if st.button("Timer", use_container_width=True,
                        type="primary" if current_state == 2 else "secondary",
                        key="power_timer_btn"):
                try:
                    api.set_device_switch_state(device_id, 2)
                    st.success("Timer mode enabled")
                    refresh_device_details(api, device_id)
                except APIError as e:
                    st.error(f"Error: {e.message}")

    with col2:
        # LED count setting
        st.markdown("**LED Count**")
        ports = details.get('ports', [])
        if ports:
            total_leds = sum(p.get('end', 0) - p.get('start', 0) + 1 for p in ports)
            st.session_state.num_leds = st.number_input(
                "Number of LEDs",
                min_value=1,
                max_value=2048,
                value=min(total_leds, 200),
                label_visibility="collapsed",
                key="led_count_input"
            )
        else:
            st.session_state.num_leds = st.number_input(
                "Number of LEDs",
                min_value=1,
                max_value=2048,
                value=st.session_state.num_leds,
                label_visibility="collapsed",
                key="led_count_input_default"
            )

    with col3:
        # View type toggle
        st.markdown("**Preview Style**")
        view_options = {"linear": "📊 Linear Strip", "house": "🏠 House Outline"}
        st.session_state.view_type = st.radio(
            "View Type",
            options=list(view_options.keys()),
            format_func=lambda x: view_options[x],
            horizontal=True,
            label_visibility="collapsed",
            key="view_type_radio"
        )


def render_builtin_effects(api: TrimlightAPI):
    """Render built-in effects selection."""
    st.subheader("🎨 Built-in Effects")

    device_id = st.session_state.selected_device
    if not device_id:
        return

    # Effect category filter
    categories = ["All"] + list(EFFECT_CATEGORIES.keys())
    selected_category = st.selectbox("Category", categories, key="builtin_category")

    # Filter effects by category
    if selected_category == "All":
        filtered_effects = BUILTIN_EFFECTS
    else:
        effect_ids = EFFECT_CATEGORIES.get(selected_category, [])
        filtered_effects = {k: v for k, v in BUILTIN_EFFECTS.items() if k in effect_ids}

    # Effect parameters
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        speed = st.slider("Speed", 0, 255, 128, key="builtin_speed")
    with col2:
        brightness = st.slider("Brightness", 0, 255, 200, key="builtin_brightness")
    with col3:
        pixel_len = st.slider("Segment Length", 1, 90, 30, key="builtin_pixel_len")
    with col4:
        reverse = st.checkbox("Reverse Direction", key="builtin_reverse")

    # Effect grid
    selected_mode = st.selectbox(
        "Select Effect",
        options=list(filtered_effects.keys()),
        format_func=lambda x: f"{x}: {filtered_effects[x]}",
        key="builtin_effect_select"
    )

    # Preview and Apply buttons
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("👁️ Preview on Device", use_container_width=True, key="builtin_preview_device"):
            try:
                api.preview_builtin_effect(
                    device_id, selected_mode, speed, brightness, pixel_len, reverse
                )
                st.success("Preview sent!")
            except APIError as e:
                st.error(f"Error: {e.message}")

    with col2:
        if st.button("🎬 Animate Preview", use_container_width=True, key="builtin_animate"):
            st.session_state.is_animating = not st.session_state.is_animating

    # Visual preview
    render_effect_preview(selected_mode, "builtin", speed, brightness, pixel_len, reverse)


def render_custom_effects(api: TrimlightAPI):
    """Render custom effects builder."""
    st.subheader("🔧 Custom Effect Builder")

    device_id = st.session_state.selected_device
    if not device_id:
        return

    # Animation mode selection
    mode = st.selectbox(
        "Animation Mode",
        options=list(CUSTOM_EFFECT_MODES.keys()),
        format_func=lambda x: CUSTOM_EFFECT_MODES[x],
        key="custom_mode_select"
    )

    col1, col2 = st.columns(2)

    with col1:
        speed = st.slider("Speed", 0, 255, 128, key="custom_speed")
    with col2:
        brightness = st.slider("Brightness", 0, 255, 200, key="custom_brightness")

    # Color segment builder
    st.markdown("**Color Segments**")

    # Initialize custom pixels if empty
    if not st.session_state.custom_pixels:
        st.session_state.custom_pixels = [
            {"index": 0, "count": 10, "color": "#ff0000", "disable": False},
            {"index": 1, "count": 10, "color": "#00ff00", "disable": False},
            {"index": 2, "count": 10, "color": "#0000ff", "disable": False},
        ]

    # Display and edit color segments
    updated_pixels = []
    for i, pixel in enumerate(st.session_state.custom_pixels):
        col1, col2, col3, col4 = st.columns([2, 2, 3, 1])

        with col1:
            count = st.number_input(
                f"Count #{i+1}",
                min_value=1,
                max_value=60,
                value=pixel.get('count', 10),
                key=f"pixel_count_{i}"
            )

        with col2:
            color = st.color_picker(
                f"Color #{i+1}",
                value=pixel.get('color', '#ffffff'),
                key=f"pixel_color_{i}"
            )

        with col3:
            disable = st.checkbox(
                f"Disable #{i+1}",
                value=pixel.get('disable', False),
                key=f"pixel_disable_{i}"
            )

        with col4:
            if st.button("🗑️", key=f"delete_pixel_{i}"):
                continue

        updated_pixels.append({
            "index": len(updated_pixels),
            "count": count,
            "color": color,
            "disable": disable
        })

    st.session_state.custom_pixels = updated_pixels

    # Add new segment button
    if st.button("➕ Add Segment", key="add_segment_btn"):
        st.session_state.custom_pixels.append({
            "index": len(st.session_state.custom_pixels),
            "count": 10,
            "color": "#ffffff",
            "disable": False
        })
        st.rerun()

    # Convert colors to API format
    api_pixels = []
    for pixel in st.session_state.custom_pixels:
        hex_color = pixel['color'].lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        api_pixels.append({
            "index": pixel['index'],
            "count": pixel['count'],
            "color": rgb_to_int(r, g, b),
            "disable": pixel['disable']
        })

    # Preview and Apply buttons
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("👁️ Preview Custom", use_container_width=True, key="custom_preview_device"):
            try:
                api.preview_custom_effect(device_id, mode, speed, brightness, api_pixels)
                st.success("Custom preview sent!")
            except APIError as e:
                st.error(f"Error: {e.message}")

    with col2:
        if st.button("🎬 Animate Custom", use_container_width=True, key="custom_animate"):
            st.session_state.is_animating = not st.session_state.is_animating

    # Visual preview
    render_effect_preview(mode, "custom", speed, brightness, pixels=api_pixels)


def render_saved_effects(api: TrimlightAPI):
    """Render saved effects list."""
    st.subheader("💾 Saved Effects")

    device_id = st.session_state.selected_device
    details = st.session_state.device_details

    if not details:
        return

    effects = details.get('effects', [])

    if not effects:
        st.info("No saved effects on this device")
        return

    for effect in effects:
        col1, col2, col3 = st.columns([3, 1, 1])

        with col1:
            effect_type = "Built-in" if effect.get('category') == 0 else "Custom"
            st.markdown(f"**{effect.get('name', 'Unnamed')}** ({effect_type})")

        with col2:
            if st.button("▶️ Play", key=f"play_{effect['id']}"):
                try:
                    api.view_effect(device_id, effect['id'])
                    st.success(f"Playing: {effect.get('name')}")
                except APIError as e:
                    st.error(f"Error: {e.message}")

        with col3:
            if st.button("🗑️", key=f"delete_{effect['id']}"):
                try:
                    api.delete_effect(device_id, effect['id'])
                    st.success("Effect deleted")
                    refresh_device_details(api, device_id)
                except APIError as e:
                    st.error(f"Error: {e.message}")


def render_effect_preview(mode: int, effect_type: str, speed: int = 100,
                          brightness: int = 200, pixel_len: int = 30,
                          reverse: bool = False, pixels: List[Dict] = None):
    """Render the visual effect preview."""
    st.markdown("---")
    st.markdown("### 👀 Visual Preview")

    num_leds = st.session_state.num_leds
    view_type = st.session_state.view_type

    renderer = EffectRenderer(num_leds)

    # Get current frame
    frame = st.session_state.preview_frame

    # Render effect
    if effect_type == "builtin":
        colors = renderer.render_builtin_effect(
            mode, frame, pixel_len, speed, brightness, reverse
        )
    else:
        colors = renderer.render_custom_effect(
            mode, frame, pixels or [], speed, brightness
        )

    # Generate and display SVG
    if view_type == "linear":
        svg = generate_linear_strip_svg(colors, width=800, height=80)
    else:
        svg = generate_house_outline_svg(colors, width=700, height=450)

    st.markdown(
        f'<div class="preview-container">{svg}</div>',
        unsafe_allow_html=True
    )

    # Animation controls
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.session_state.is_animating:
            st.session_state.preview_frame += 1
            time.sleep(0.05)
            st.rerun()

    with col2:
        st.markdown(f"**Frame:** {frame}")

    with col3:
        if st.button("⏮️ Reset Frame", key=f"reset_frame_{effect_type}"):
            st.session_state.preview_frame = 0
            st.rerun()


def render_schedules(api: TrimlightAPI):
    """Render schedule management."""
    st.subheader("📅 Schedules")

    details = st.session_state.device_details
    if not details:
        return

    # Daily schedules
    st.markdown("**Daily Schedules**")
    daily = details.get('daily', [])

    for schedule in daily:
        with st.expander(f"Daily Schedule {schedule.get('id', 0) + 1}"):
            enabled = st.checkbox(
                "Enabled",
                value=schedule.get('enable', False),
                key=f"daily_enable_{schedule['id']}"
            )

            col1, col2 = st.columns(2)
            with col1:
                start_time = st.time_input(
                    "Start Time",
                    value=None,
                    key=f"daily_start_{schedule['id']}"
                )
            with col2:
                end_time = st.time_input(
                    "End Time",
                    value=None,
                    key=f"daily_end_{schedule['id']}"
                )

    # Calendar schedules
    st.markdown("**Calendar Schedules**")
    calendar = details.get('calendar', [])

    if calendar:
        for schedule in calendar[:5]:  # Show first 5
            st.markdown(
                f"- Effect {schedule.get('effectId')}: "
                f"{schedule.get('startDate', {}).get('month')}/{schedule.get('startDate', {}).get('day')} - "
                f"{schedule.get('endDate', {}).get('month')}/{schedule.get('endDate', {}).get('day')}"
            )
    else:
        st.info("No calendar schedules configured")


def main():
    """Main application entry point."""
    init_session_state()

    # Header
    st.title("💡 Trimlight LED Controller")
    st.markdown("Configure and preview your Trimlight home exterior LED lights")

    # Get API client
    api = get_or_create_api_client()

    if api is None:
        st.error("⚠️ API credentials not configured!")
        st.markdown("""
        ### Setup Instructions

        Add your Trimlight credentials to Streamlit secrets:

        1. In Streamlit Cloud, go to your app settings
        2. Click on "Secrets" in the left menu
        3. Add the following:

        ```toml
        [trimlight]
        client_id = "your_client_id"
        client_secret = "your_client_secret"
        ```

        Or as environment variables:

        ```toml
        TRIMLIGHT_CLIENT_ID = "your_client_id"
        TRIMLIGHT_CLIENT_SECRET = "your_client_secret"
        ```
        """)
        return

    # Render device selector in sidebar
    render_device_selector(api)

    # Main content
    if st.session_state.selected_device:
        # Device controls
        render_device_controls(api)

        st.markdown("---")

        # Tabs for different features
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎨 Built-in Effects",
            "🔧 Custom Effects",
            "💾 Saved Effects",
            "📅 Schedules"
        ])

        with tab1:
            render_builtin_effects(api)

        with tab2:
            render_custom_effects(api)

        with tab3:
            render_saved_effects(api)

        with tab4:
            render_schedules(api)

    else:
        st.info("👈 Select a device from the sidebar to get started")

        # Demo preview without device
        st.markdown("---")
        st.subheader("🎬 Demo Preview")

        demo_col1, demo_col2 = st.columns(2)

        with demo_col1:
            demo_mode = st.selectbox(
                "Demo Effect",
                options=list(BUILTIN_EFFECTS.keys())[:20],
                format_func=lambda x: BUILTIN_EFFECTS[x],
                key="demo_effect_select"
            )

        with demo_col2:
            demo_leds = st.slider("LED Count", 20, 200, 100, key="demo_led_slider")

        st.session_state.num_leds = demo_leds

        # Render demo preview
        renderer = EffectRenderer(demo_leds)
        colors = renderer.render_builtin_effect(
            demo_mode, st.session_state.preview_frame, 30, 128, 200, False
        )

        svg_linear = generate_linear_strip_svg(colors, width=800, height=80)
        svg_house = generate_house_outline_svg(colors, width=700, height=400)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Linear Strip View**")
            st.markdown(f'<div class="preview-container">{svg_linear}</div>',
                       unsafe_allow_html=True)

        with col2:
            st.markdown("**House Outline View**")
            st.markdown(f'<div class="preview-container">{svg_house}</div>',
                       unsafe_allow_html=True)

        # Auto-animate demo
        if st.checkbox("▶️ Animate Demo", key="demo_animate_checkbox"):
            st.session_state.preview_frame += 1
            time.sleep(0.05)
            st.rerun()


if __name__ == "__main__":
    main()
