"""
LED Visualization Components
Provides visual representations of LED light configurations.
"""

import math
from typing import List, Dict, Tuple, Optional
import colorsys


def int_to_rgb(color_int: int) -> Tuple[int, int, int]:
    """Convert integer color value to RGB tuple."""
    r = (color_int >> 16) & 0xFF
    g = (color_int >> 8) & 0xFF
    b = color_int & 0xFF
    return (r, g, b)


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """Convert RGB values to hex string."""
    return f"#{r:02x}{g:02x}{b:02x}"


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color string to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_int(r: int, g: int, b: int) -> int:
    """Convert RGB tuple to integer color value."""
    return (r << 16) | (g << 8) | b


def blend_colors(color1: Tuple[int, int, int], color2: Tuple[int, int, int],
                 factor: float) -> Tuple[int, int, int]:
    """Blend two colors together. Factor 0 = color1, 1 = color2."""
    r = int(color1[0] + (color2[0] - color1[0]) * factor)
    g = int(color1[1] + (color2[1] - color1[1]) * factor)
    b = int(color1[2] + (color2[2] - color1[2]) * factor)
    return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))


def get_rainbow_color(position: float) -> Tuple[int, int, int]:
    """Get rainbow color at position (0-1)."""
    h = position % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
    return (int(r * 255), int(g * 255), int(b * 255))


def apply_brightness(color: Tuple[int, int, int], brightness: int) -> Tuple[int, int, int]:
    """Apply brightness (0-255) to a color."""
    factor = brightness / 255.0
    return (
        int(color[0] * factor),
        int(color[1] * factor),
        int(color[2] * factor)
    )


class LEDStrip:
    """Represents an LED strip with color values."""

    def __init__(self, num_leds: int):
        self.num_leds = num_leds
        self.colors: List[Tuple[int, int, int]] = [(0, 0, 0)] * num_leds

    def set_all(self, color: Tuple[int, int, int]):
        """Set all LEDs to the same color."""
        self.colors = [color] * self.num_leds

    def set_led(self, index: int, color: Tuple[int, int, int]):
        """Set a specific LED color."""
        if 0 <= index < self.num_leds:
            self.colors[index] = color

    def set_range(self, start: int, end: int, color: Tuple[int, int, int]):
        """Set a range of LEDs to the same color."""
        for i in range(max(0, start), min(end, self.num_leds)):
            self.colors[i] = color

    def clear(self):
        """Turn off all LEDs."""
        self.colors = [(0, 0, 0)] * self.num_leds


class EffectRenderer:
    """Renders various LED effects for visualization."""

    # Standard colors
    COLORS = {
        'red': (255, 0, 0),
        'green': (0, 255, 0),
        'blue': (0, 0, 255),
        'yellow': (255, 255, 0),
        'cyan': (0, 255, 255),
        'purple': (128, 0, 255),
        'white': (255, 255, 255),
        'off': (0, 0, 0)
    }

    def __init__(self, num_leds: int):
        self.num_leds = num_leds
        self.strip = LEDStrip(num_leds)

    def render_static(self, pixels: List[Dict], brightness: int = 255) -> List[Tuple[int, int, int]]:
        """Render a static custom pattern."""
        self.strip.clear()

        current_pos = 0
        for pixel in pixels:
            if pixel.get('disable', False):
                current_pos += pixel.get('count', 1)
                continue

            color = int_to_rgb(pixel.get('color', 0))
            color = apply_brightness(color, brightness)
            count = pixel.get('count', 1)

            for i in range(count):
                if current_pos < self.num_leds:
                    self.strip.set_led(current_pos, color)
                    current_pos += 1

        return self.strip.colors

    def render_builtin_effect(self, mode: int, frame: int, pixel_len: int = 30,
                              speed: int = 100, brightness: int = 255,
                              reverse: bool = False) -> List[Tuple[int, int, int]]:
        """Render a frame of a built-in effect."""
        self.strip.clear()

        # Normalize frame based on speed (higher speed = faster animation)
        speed_factor = speed / 100.0
        animated_frame = int(frame * speed_factor)

        if reverse:
            animated_frame = -animated_frame

        # Rainbow effects (0-9)
        if mode < 10:
            self._render_rainbow_effect(mode, animated_frame, pixel_len, brightness)
        # Stacking effects (10-23)
        elif mode < 24:
            self._render_stacking_effect(mode, animated_frame, pixel_len, brightness)
        # Comet effects (24-30)
        elif mode < 31:
            self._render_comet_effect(mode, animated_frame, pixel_len, brightness)
        # Meteor effects (31-37)
        elif mode < 38:
            self._render_meteor_effect(mode, animated_frame, pixel_len, brightness)
        # Wave effects (38-65)
        elif mode < 66:
            self._render_wave_effect(mode, animated_frame, pixel_len, brightness)
        # Pulse effects (66-84)
        elif mode < 85:
            self._render_pulse_effect(mode, animated_frame, pixel_len, brightness)
        # Spin effects (85-105)
        elif mode < 106:
            self._render_spin_effect(mode, animated_frame, pixel_len, brightness)
        # Snake effects (106-137)
        elif mode < 138:
            self._render_snake_effect(mode, animated_frame, pixel_len, brightness)
        # Stars effects (138-157)
        elif mode < 158:
            self._render_stars_effect(mode, animated_frame, pixel_len, brightness)
        # Breath effects (158-164)
        elif mode < 165:
            self._render_breath_effect(mode, animated_frame, pixel_len, brightness)
        # Fire effects (165-170)
        elif mode < 171:
            self._render_fire_effect(mode, animated_frame, pixel_len, brightness)
        # Strobe effects (171-179)
        else:
            self._render_strobe_effect(mode, animated_frame, pixel_len, brightness)

        return self.strip.colors

    def _render_rainbow_effect(self, mode: int, frame: int, pixel_len: int, brightness: int):
        """Render rainbow-based effects."""
        for i in range(self.num_leds):
            if mode == 0:  # Rainbow Gradual Chase
                pos = (i + frame) % self.num_leds
                color = get_rainbow_color(pos / pixel_len)
            elif mode == 1:  # Rainbow Comet
                tail_len = pixel_len
                head_pos = frame % self.num_leds
                dist = (head_pos - i) % self.num_leds
                if dist < tail_len:
                    intensity = 1.0 - (dist / tail_len)
                    color = get_rainbow_color(i / pixel_len)
                    color = apply_brightness(color, int(brightness * intensity))
                    self.strip.set_led(i, color)
                continue
            elif mode == 2:  # Rainbow Segment
                segment = ((i + frame) // pixel_len) % 7
                color = get_rainbow_color(segment / 7)
            elif mode == 3:  # Rainbow Wave
                wave = math.sin((i + frame) * 2 * math.pi / pixel_len)
                intensity = (wave + 1) / 2
                color = get_rainbow_color(i / self.num_leds)
                color = apply_brightness(color, int(brightness * intensity))
                self.strip.set_led(i, color)
                continue
            elif mode == 4:  # Rainbow Meteor
                meteor_pos = frame % (self.num_leds + pixel_len)
                dist = meteor_pos - i
                if 0 <= dist < pixel_len:
                    intensity = 1.0 - (dist / pixel_len)
                    color = get_rainbow_color(i / pixel_len)
                    color = apply_brightness(color, int(brightness * intensity))
                    self.strip.set_led(i, color)
                continue
            elif mode == 5:  # Rainbow Gradual
                phase = (frame / 50) % 1.0
                color = get_rainbow_color((i / self.num_leds + phase) % 1.0)
            elif mode == 6:  # Rainbow Jump
                segment = (frame // 10) % 7
                color = get_rainbow_color(segment / 7)
            elif mode == 7:  # Rainbow Stars
                if (i + frame) % pixel_len == 0:
                    color = get_rainbow_color(i / self.num_leds)
                else:
                    color = (0, 0, 0)
            elif mode == 8:  # Rainbow Fade In Out
                phase = abs(math.sin(frame * math.pi / 60))
                color = get_rainbow_color(i / self.num_leds)
                color = apply_brightness(color, int(brightness * phase))
                self.strip.set_led(i, color)
                continue
            elif mode == 9:  # Rainbow Spin
                angle = (i / self.num_leds + frame / 50) % 1.0
                color = get_rainbow_color(angle)
            else:
                color = get_rainbow_color(i / self.num_leds)

            color = apply_brightness(color, brightness)
            self.strip.set_led(i, color)

    def _get_color_for_mode(self, mode: int, base_offset: int) -> Tuple[int, int, int]:
        """Get the primary color for a given effect mode."""
        color_map = [
            self.COLORS['red'], self.COLORS['green'], self.COLORS['blue'],
            self.COLORS['yellow'], self.COLORS['cyan'], self.COLORS['purple'],
            self.COLORS['white']
        ]
        index = (mode - base_offset) % len(color_map)
        return color_map[index]

    def _render_stacking_effect(self, mode: int, frame: int, pixel_len: int, brightness: int):
        """Render stacking effects."""
        color = self._get_color_for_mode(mode, 10)
        stack_height = (frame // 2) % self.num_leds

        for i in range(stack_height):
            self.strip.set_led(i, apply_brightness(color, brightness))

    def _render_comet_effect(self, mode: int, frame: int, pixel_len: int, brightness: int):
        """Render comet effects."""
        color = self._get_color_for_mode(mode, 24)
        tail_len = pixel_len
        head_pos = frame % (self.num_leds + tail_len)

        for i in range(self.num_leds):
            dist = head_pos - i
            if 0 <= dist < tail_len:
                intensity = 1.0 - (dist / tail_len)
                self.strip.set_led(i, apply_brightness(color, int(brightness * intensity)))

    def _render_meteor_effect(self, mode: int, frame: int, pixel_len: int, brightness: int):
        """Render meteor effects."""
        color = self._get_color_for_mode(mode, 31)
        meteor_len = pixel_len
        meteor_pos = frame % (self.num_leds + meteor_len)

        for i in range(self.num_leds):
            dist = meteor_pos - i
            if 0 <= dist < meteor_len:
                intensity = (1.0 - (dist / meteor_len)) ** 2
                self.strip.set_led(i, apply_brightness(color, int(brightness * intensity)))

    def _render_wave_effect(self, mode: int, frame: int, pixel_len: int, brightness: int):
        """Render wave effects."""
        if mode < 45:  # Single color waves
            color = self._get_color_for_mode(mode, 38)
            color2 = (0, 0, 0)
        else:  # Two color waves
            colors = [
                (self.COLORS['red'], self.COLORS['green']),
                (self.COLORS['red'], self.COLORS['blue']),
                (self.COLORS['red'], self.COLORS['yellow']),
                (self.COLORS['red'], self.COLORS['cyan']),
                (self.COLORS['red'], self.COLORS['purple']),
                (self.COLORS['red'], self.COLORS['white']),
                (self.COLORS['green'], self.COLORS['blue']),
                (self.COLORS['green'], self.COLORS['yellow']),
                (self.COLORS['green'], self.COLORS['cyan']),
                (self.COLORS['green'], self.COLORS['purple']),
                (self.COLORS['green'], self.COLORS['white']),
                (self.COLORS['blue'], self.COLORS['yellow']),
                (self.COLORS['blue'], self.COLORS['cyan']),
                (self.COLORS['blue'], self.COLORS['purple']),
                (self.COLORS['blue'], self.COLORS['white']),
                (self.COLORS['yellow'], self.COLORS['cyan']),
                (self.COLORS['yellow'], self.COLORS['purple']),
                (self.COLORS['yellow'], self.COLORS['white']),
                (self.COLORS['cyan'], self.COLORS['purple']),
                (self.COLORS['cyan'], self.COLORS['white']),
                (self.COLORS['purple'], self.COLORS['white']),
            ]
            idx = (mode - 45) % len(colors)
            color, color2 = colors[idx]

        for i in range(self.num_leds):
            wave = math.sin((i + frame) * 2 * math.pi / pixel_len)
            if mode < 45:
                intensity = (wave + 1) / 2
                self.strip.set_led(i, apply_brightness(color, int(brightness * intensity)))
            else:
                factor = (wave + 1) / 2
                blended = blend_colors(color, color2, factor)
                self.strip.set_led(i, apply_brightness(blended, brightness))

    def _render_pulse_effect(self, mode: int, frame: int, pixel_len: int, brightness: int):
        """Render pulse effects."""
        color = self._get_color_for_mode(mode, 66)
        pulse_pos = frame % self.num_leds
        pulse_width = pixel_len // 2

        for i in range(self.num_leds):
            dist = abs(i - pulse_pos)
            if dist < pulse_width:
                intensity = 1.0 - (dist / pulse_width)
                self.strip.set_led(i, apply_brightness(color, int(brightness * intensity)))

    def _render_spin_effect(self, mode: int, frame: int, pixel_len: int, brightness: int):
        """Render spin effects."""
        color = self._get_color_for_mode(mode, 85)
        segment_size = pixel_len
        rotation = frame % self.num_leds

        for i in range(self.num_leds):
            pos = (i + rotation) % self.num_leds
            if (pos // segment_size) % 2 == 0:
                self.strip.set_led(i, apply_brightness(color, brightness))

    def _render_snake_effect(self, mode: int, frame: int, pixel_len: int, brightness: int):
        """Render snake effects."""
        # Two-color gradual snake
        colors = [
            (self.COLORS['red'], self.COLORS['green']),
            (self.COLORS['red'], self.COLORS['blue']),
            (self.COLORS['red'], self.COLORS['yellow']),
        ]
        idx = (mode - 106) % len(colors)
        color1, color2 = colors[min(idx, len(colors)-1)]

        snake_len = pixel_len
        head_pos = frame % (self.num_leds + snake_len)

        for i in range(self.num_leds):
            dist = head_pos - i
            if 0 <= dist < snake_len:
                factor = dist / snake_len
                blended = blend_colors(color1, color2, factor)
                intensity = 1.0 - (dist / snake_len) * 0.5
                self.strip.set_led(i, apply_brightness(blended, int(brightness * intensity)))

    def _render_stars_effect(self, mode: int, frame: int, pixel_len: int, brightness: int):
        """Render stars/twinkle effects."""
        color = self._get_color_for_mode(mode, 138)

        # Background
        if mode >= 145:
            bg_color = apply_brightness(color, brightness // 4)
            self.strip.set_all(bg_color)

        # Twinkling stars
        import random
        random.seed(frame // 5)  # Change stars every 5 frames
        num_stars = self.num_leds // pixel_len

        for _ in range(num_stars):
            pos = random.randint(0, self.num_leds - 1)
            twinkle = random.random()
            self.strip.set_led(pos, apply_brightness(color, int(brightness * twinkle)))

    def _render_breath_effect(self, mode: int, frame: int, pixel_len: int, brightness: int):
        """Render breathing effects."""
        color = self._get_color_for_mode(mode, 158)

        # Smooth breathing cycle
        phase = (math.sin(frame * math.pi / 30) + 1) / 2
        intensity = 0.1 + 0.9 * phase

        breath_color = apply_brightness(color, int(brightness * intensity))
        self.strip.set_all(breath_color)

    def _render_fire_effect(self, mode: int, frame: int, pixel_len: int, brightness: int):
        """Render fire effects."""
        import random
        random.seed(frame)

        # Fire colors based on mode
        if mode in [165, 167]:  # Red-Yellow or Green-Yellow
            base = self.COLORS['yellow'] if mode == 165 else self.COLORS['green']
            tip = self.COLORS['red'] if mode == 165 else self.COLORS['yellow']
        else:
            base = self.COLORS['cyan']
            tip = self.COLORS['blue']

        for i in range(self.num_leds):
            flicker = random.random() * 0.4 + 0.6
            height_factor = i / self.num_leds
            color = blend_colors(base, tip, height_factor)
            self.strip.set_led(i, apply_brightness(color, int(brightness * flicker)))

    def _render_strobe_effect(self, mode: int, frame: int, pixel_len: int, brightness: int):
        """Render strobe effects."""
        color = self._get_color_for_mode(mode, 171)

        # Fast on/off strobe
        if (frame // 3) % 2 == 0:
            self.strip.set_all(apply_brightness(color, brightness))
        else:
            self.strip.clear()

    def render_custom_effect(self, mode: int, frame: int, pixels: List[Dict],
                             speed: int = 100, brightness: int = 255) -> List[Tuple[int, int, int]]:
        """Render a custom effect with animation."""
        # First render the base pattern
        base_colors = self.render_static(pixels, brightness)

        speed_factor = speed / 100.0
        animated_frame = int(frame * speed_factor)

        if mode == 0:  # Static
            return base_colors
        elif mode in [1, 2]:  # Chase Forward/Backward
            direction = 1 if mode == 1 else -1
            shift = (animated_frame * direction) % self.num_leds
            return base_colors[shift:] + base_colors[:shift]
        elif mode in [3, 4]:  # Chase Middle to Out / Out to Middle
            mid = self.num_leds // 2
            expansion = animated_frame % mid
            result = [(0, 0, 0)] * self.num_leds
            if mode == 3:
                for i in range(expansion):
                    if mid - i - 1 >= 0:
                        result[mid - i - 1] = base_colors[mid - i - 1]
                    if mid + i < self.num_leds:
                        result[mid + i] = base_colors[mid + i]
            else:
                for i in range(mid - expansion, mid + expansion):
                    if 0 <= i < self.num_leds:
                        result[i] = base_colors[i]
            return result
        elif mode == 5:  # Stars
            import random
            random.seed(animated_frame // 5)
            result = [(0, 0, 0)] * self.num_leds
            for i, color in enumerate(base_colors):
                if color != (0, 0, 0) and random.random() > 0.5:
                    result[i] = color
            return result
        elif mode == 6:  # Breath
            phase = (math.sin(animated_frame * math.pi / 30) + 1) / 2
            intensity = 0.1 + 0.9 * phase
            return [apply_brightness(c, int(255 * intensity)) for c in base_colors]
        elif mode in [7, 8, 9, 10]:  # Comet variations
            tail_len = 10
            head_pos = animated_frame % (self.num_leds + tail_len)
            result = [(0, 0, 0)] * self.num_leds
            for i in range(self.num_leds):
                dist = head_pos - i if mode in [7, 9] else i - (self.num_leds - head_pos)
                if 0 <= dist < tail_len:
                    intensity = 1.0 - (dist / tail_len)
                    result[i] = apply_brightness(base_colors[i % len(base_colors)],
                                                 int(brightness * intensity))
            return result
        elif mode in [11, 12, 13, 14]:  # Wave variations
            result = []
            for i in range(self.num_leds):
                wave = math.sin((i + animated_frame) * 2 * math.pi / 20)
                intensity = (wave + 1) / 2
                result.append(apply_brightness(base_colors[i], int(brightness * intensity)))
            return result
        elif mode == 15:  # Strobe
            if (animated_frame // 3) % 2 == 0:
                return base_colors
            else:
                return [(0, 0, 0)] * self.num_leds
        elif mode == 16:  # Solid Fade
            total_colors = []
            for p in pixels:
                if not p.get('disable', False):
                    total_colors.append(int_to_rgb(p.get('color', 0)))
            if not total_colors:
                return base_colors
            color_idx = (animated_frame // 30) % len(total_colors)
            next_idx = (color_idx + 1) % len(total_colors)
            factor = (animated_frame % 30) / 30
            blended = blend_colors(total_colors[color_idx], total_colors[next_idx], factor)
            return [apply_brightness(blended, brightness)] * self.num_leds

        return base_colors


def generate_linear_strip_svg(colors: List[Tuple[int, int, int]],
                              width: int = 800, height: int = 60,
                              led_spacing: int = 4) -> str:
    """Generate SVG for linear LED strip visualization."""
    num_leds = len(colors)
    if num_leds == 0:
        return ""

    # Calculate LED size based on number of LEDs
    available_width = width - 20
    led_width = max(2, min(20, (available_width - (num_leds - 1) * led_spacing) // num_leds))
    total_strip_width = num_leds * led_width + (num_leds - 1) * led_spacing
    start_x = (width - total_strip_width) // 2

    svg_parts = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
        f'<rect width="{width}" height="{height}" fill="#1a1a2e"/>',
        # Strip background
        f'<rect x="{start_x - 5}" y="{height//2 - 15}" width="{total_strip_width + 10}" '
        f'height="30" rx="5" fill="#2d2d44"/>',
    ]

    # Draw LEDs
    for i, color in enumerate(colors):
        x = start_x + i * (led_width + led_spacing)
        r, g, b = color

        # LED glow effect
        if r > 10 or g > 10 or b > 10:
            glow_color = f"rgba({r},{g},{b},0.5)"
            svg_parts.append(
                f'<rect x="{x-2}" y="{height//2 - 12}" width="{led_width+4}" height="24" '
                f'rx="3" fill="{glow_color}" filter="url(#glow)"/>'
            )

        # LED body
        hex_color = rgb_to_hex(r, g, b)
        svg_parts.append(
            f'<rect x="{x}" y="{height//2 - 10}" width="{led_width}" height="20" '
            f'rx="2" fill="{hex_color}" stroke="#444" stroke-width="0.5"/>'
        )

    # Add glow filter definition
    svg_parts.insert(1, '''
        <defs>
            <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
                <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
                <feMerge>
                    <feMergeNode in="coloredBlur"/>
                    <feMergeNode in="SourceGraphic"/>
                </feMerge>
            </filter>
        </defs>
    ''')

    svg_parts.append('</svg>')
    return ''.join(svg_parts)


def generate_house_outline_svg(colors: List[Tuple[int, int, int]],
                                width: int = 600, height: int = 400) -> str:
    """Generate SVG for house outline LED visualization."""
    num_leds = len(colors)
    if num_leds == 0:
        return ""

    # House dimensions
    house_width = width * 0.7
    house_height = height * 0.5
    roof_height = height * 0.25
    start_x = (width - house_width) / 2
    base_y = height * 0.85
    roof_peak_y = height * 0.15
    roof_edge_y = base_y - house_height

    # Calculate path points for roofline
    path_points = [
        (start_x, roof_edge_y),  # Left roof edge
        (width / 2, roof_peak_y),  # Roof peak
        (start_x + house_width, roof_edge_y),  # Right roof edge
    ]

    # Calculate total path length
    def distance(p1, p2):
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

    segment_lengths = [
        distance(path_points[0], path_points[1]),
        distance(path_points[1], path_points[2])
    ]
    total_length = sum(segment_lengths)

    # Place LEDs along the path
    led_positions = []
    for i in range(num_leds):
        progress = i / max(1, num_leds - 1)
        dist_along = progress * total_length

        if dist_along <= segment_lengths[0]:
            # On first segment (left side of roof)
            t = dist_along / segment_lengths[0]
            x = path_points[0][0] + t * (path_points[1][0] - path_points[0][0])
            y = path_points[0][1] + t * (path_points[1][1] - path_points[0][1])
        else:
            # On second segment (right side of roof)
            remaining = dist_along - segment_lengths[0]
            t = remaining / segment_lengths[1]
            x = path_points[1][0] + t * (path_points[2][0] - path_points[1][0])
            y = path_points[1][1] + t * (path_points[2][1] - path_points[1][1])

        led_positions.append((x, y))

    # Generate SVG
    svg_parts = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
        # Night sky background
        f'<rect width="{width}" height="{height}" fill="#0a0a1a"/>',
        # Stars
    ]

    # Add some stars
    import random
    random.seed(42)
    for _ in range(30):
        sx = random.randint(10, width - 10)
        sy = random.randint(10, int(height * 0.6))
        size = random.random() * 1.5 + 0.5
        svg_parts.append(f'<circle cx="{sx}" cy="{sy}" r="{size}" fill="#ffffff" opacity="0.6"/>')

    # House body (darker)
    svg_parts.append(
        f'<rect x="{start_x}" y="{roof_edge_y}" width="{house_width}" '
        f'height="{house_height}" fill="#1a1a2a"/>'
    )

    # Roof
    roof_path = f'M {start_x - 20} {roof_edge_y} L {width/2} {roof_peak_y} L {start_x + house_width + 20} {roof_edge_y} Z'
    svg_parts.append(f'<path d="{roof_path}" fill="#252535" stroke="#333" stroke-width="2"/>')

    # Windows
    window_width = house_width * 0.15
    window_height = house_height * 0.25
    window_y = roof_edge_y + house_height * 0.2

    for wx in [start_x + house_width * 0.15, start_x + house_width * 0.55]:
        svg_parts.append(
            f'<rect x="{wx}" y="{window_y}" width="{window_width}" height="{window_height}" '
            f'fill="#2a2a3a" stroke="#3a3a4a" stroke-width="2"/>'
        )

    # Door
    door_width = house_width * 0.15
    door_height = house_height * 0.4
    door_x = start_x + house_width * 0.42
    door_y = base_y - door_height
    svg_parts.append(
        f'<rect x="{door_x}" y="{door_y}" width="{door_width}" height="{door_height}" '
        f'fill="#2a2a3a" stroke="#3a3a4a" stroke-width="2"/>'
    )

    # Add glow filter
    svg_parts.append('''
        <defs>
            <filter id="ledglow" x="-100%" y="-100%" width="300%" height="300%">
                <feGaussianBlur stdDeviation="4" result="coloredBlur"/>
                <feMerge>
                    <feMergeNode in="coloredBlur"/>
                    <feMergeNode in="SourceGraphic"/>
                </feMerge>
            </filter>
        </defs>
    ''')

    # Draw LEDs along roofline
    led_size = max(3, min(8, 300 / num_leds))

    for i, ((x, y), color) in enumerate(zip(led_positions, colors)):
        r, g, b = color

        # LED glow
        if r > 10 or g > 10 or b > 10:
            glow_color = f"rgba({r},{g},{b},0.7)"
            svg_parts.append(
                f'<circle cx="{x}" cy="{y}" r="{led_size * 2}" '
                f'fill="{glow_color}" filter="url(#ledglow)"/>'
            )

        # LED point
        hex_color = rgb_to_hex(r, g, b)
        svg_parts.append(
            f'<circle cx="{x}" cy="{y}" r="{led_size}" fill="{hex_color}"/>'
        )

    svg_parts.append('</svg>')
    return ''.join(svg_parts)


def generate_preview_html(colors: List[Tuple[int, int, int]], view_type: str = "linear") -> str:
    """Generate HTML with embedded SVG for preview."""
    if view_type == "linear":
        svg = generate_linear_strip_svg(colors)
    else:
        svg = generate_house_outline_svg(colors)

    return svg
