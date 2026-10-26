"""Procedural views drawn from public observations, never privileged task state."""

from math import cos, pi, sin, sqrt

from PIL import Image, ImageDraw, ImageFont

from panel_guide.engine import MazeView, RoomView
from panel_guide.tasks import SYMBOLS

INK = "#20313e"
BACKGROUND = "#eef2f4"
ROOM_RENDER_VERSION = "v2-leaf"
RENDER_VERSION = "v4-maze-min-width"
DETAIL_RENDER_VERSION = "v5-maze-overview-detail"
ROOM_SIZE, ROOM_GAP, MARGIN, TOP = 280, 88, 48, 110


def render_version(observation_mode: str) -> str:
    if observation_mode == "maze_detail":
        return DETAIL_RENDER_VERSION
    if observation_mode == "maze":
        return RENDER_VERSION
    if observation_mode == "room":
        return ROOM_RENDER_VERSION
    raise ValueError("observation_mode must be maze_detail, maze, or room.")


def room_center(view: MazeView, position: tuple[int, int]) -> tuple[int, int]:
    min_x = min(r.position[0] for r in view.rooms)
    min_y = min(r.position[1] for r in view.rooms)
    step = ROOM_SIZE + ROOM_GAP
    return (MARGIN + (position[0] - min_x) * step + ROOM_SIZE // 2,
            TOP + (position[1] - min_y) * step + ROOM_SIZE // 2)


def room_crop_bounds(view: MazeView, image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    """Public current-location marker selects the crop; no hidden task fields.

    Padding includes local door IDs, corridor gates, and their OPEN/LOCKED labels.
    It stops before neighboring room emblems and omits the overview's header.
    """
    current = [room for room in view.rooms if room.current]
    if len(current) != 1:
        raise ValueError("A detail crop requires exactly one current room.")
    x, y = room_center(view, current[0].position)
    half = ROOM_SIZE // 2 + 104
    width, height = image_size
    return (max(MARGIN - 12, x - half), max(TOP - 12, y - half),
            min(width - MARGIN + 12, x + half), min(height - 94 + 12, y + half))


def crop_current_room(overview: Image.Image, view: MazeView) -> Image.Image:
    """Enlarge pixels from this exact redacted screenshot; reveal no new state."""
    with overview.crop(room_crop_bounds(view, overview.size)) as crop:
        return crop.resize((crop.width * 2, crop.height * 2), Image.Resampling.NEAREST)


def draw_symbol(draw: ImageDraw.ImageDraw, symbol: str, center: tuple[float, float],
                radius: float, background: str = "white") -> None:
    if symbol not in SYMBOLS:
        raise ValueError(f"Unsupported symbol: {symbol}")
    x, y = center
    r = radius
    box = (x - r, y - r, x + r, y + r)
    width = max(3, round(r / 9))

    def polygon(points):
        draw.polygon([(x + dx * r, y + dy * r) for dx, dy in points], fill=INK)

    if symbol in ("ring", "solid circle", "striped circle"):
        draw.ellipse(box, fill=INK if symbol == "solid circle" else background, outline=INK, width=width)
        if symbol == "striped circle":
            for fraction in (-0.6, -0.3, 0, 0.3, 0.6):
                dx = fraction * r
                dy = sqrt((r - width) ** 2 - dx ** 2)
                draw.line((x + dx, y - dy, x + dx, y + dy), fill=INK, width=width)
    elif symbol == "bars":
        for offset in (-0.6, 0, 0.6):
            draw.line((x - r, y + offset * r, x + r, y + offset * r), fill=INK, width=width * 2)
    elif symbol == "plus":
        draw.line((x - r, y, x + r, y), fill=INK, width=width * 2)
        draw.line((x, y - r, x, y + r), fill=INK, width=width * 2)
    elif symbol == "triangle":
        polygon([(0, -1), (1, 0.85), (-1, 0.85)])
    elif symbol == "diamond":
        polygon([(0, -1), (1, 0), (0, 1), (-1, 0)])
    elif symbol == "zigzag":
        points = [(-1, -0.7), (-0.35, -0.7), (-0.75, 0), (0.6, 0), (0.2, 0.7), (1, 0.7)]
        draw.line([(x + dx * r, y + dy * r) for dx, dy in points], fill=INK, width=width * 2, joint="curve")
    elif symbol == "crescent":
        draw.ellipse(box, fill=INK)
        draw.ellipse((x - 0.35 * r, y - 1.1 * r, x + 1.35 * r, y + 0.6 * r), fill=background)
    elif symbol == "sun":
        draw.ellipse((x - r * 0.55, y - r * 0.55, x + r * 0.55, y + r * 0.55), fill=INK)
        for index in range(8):
            angle = index * pi / 4
            draw.line((x + cos(angle) * r * 0.75, y + sin(angle) * r * 0.75,
                       x + cos(angle) * r, y + sin(angle) * r), fill=INK, width=width)
    elif symbol == "star":
        points = []
        for index in range(10):
            angle = -pi / 2 + index * pi / 5
            size = r if index % 2 == 0 else r * 0.43
            points.append((x + cos(angle) * size, y + sin(angle) * size))
        draw.polygon(points, fill=INK)
    elif symbol == "leaf":
        # Curved pointed blade, stem, and paired veins instead of a polygon/T.
        points = []
        for side in (1, -1):
            samples = range(25) if side == 1 else reversed(range(25))
            for index in samples:
                t = index / 24
                px = side * (3 * (1 - t) ** 2 * t * 0.95 + 3 * (1 - t) * t ** 2 * 0.85)
                py = (1 - t) ** 3 * -1 + 3 * (1 - t) ** 2 * t * -0.55 + 3 * (1 - t) * t ** 2 * 0.55 + t ** 3 * 0.8
                points.append((px, py))
        draw.line((x, y + r * 0.65, x, y + r * 1.1), fill=INK, width=width)
        polygon(points)
        draw.line((x, y - r * 0.8, x, y + r * 0.65), fill=background, width=max(2, width - 1))
        for height, reach in ((-0.35, 0.36), (0, 0.52), (0.35, 0.38)):
            for side in (-1, 1):
                draw.line((x, y + height * r, x + side * reach * r, y + (height - 0.25) * r),
                          fill=background, width=max(2, width - 1))
    elif symbol == "anvil":
        polygon([(-1, -0.6), (1, -0.6), (0.75, -0.15), (0.3, -0.15),
                 (0.3, 0.3), (0.8, 0.7), (-0.8, 0.7), (-0.3, 0.3),
                 (-0.3, -0.15), (-0.75, -0.15)])


def render_room(view: RoomView) -> Image.Image:
    image = Image.new("RGB", (960, 720), BACKGROUND)
    draw = ImageDraw.Draw(image)
    label = ImageFont.load_default(size=24)
    small = ImageFont.load_default(size=19)
    draw.text((480, 30), "ROOM EMBLEM", font=small, fill=INK, anchor="mm")
    draw_symbol(draw, view.emblem, (480, 106), 46, BACKGROUND)
    draw.line((48, 180, 912, 180), fill="#c8d2d9", width=2)

    width, gap = 232, 32
    total = len(view.doors) * width + (len(view.doors) - 1) * gap
    for index, door in enumerate(view.doors):
        x = (960 - total) / 2 + index * (width + gap)
        draw.rounded_rectangle((x, 210, x + width, 500), radius=18, fill="white", outline="#b8c6cf", width=2)
        draw.text((x + width / 2, 238), door.id, font=label, fill=INK, anchor="mm")
        draw_symbol(draw, door.symbol, (x + width / 2, 304), 32)
        draw.rounded_rectangle((x + 69, 356, x + 163, 432), radius=6, fill="#e6edf1", outline=INK, width=3)
        draw.ellipse((x + 142, 391, x + 150, 399), fill=INK)
        color = "#17654e" if door.is_open else "#8b3a2a"
        draw.rounded_rectangle((x + 40, 451, x + width - 40, 482), radius=8, fill=color)
        draw.text((x + width / 2, 466), "OPEN" if door.is_open else "LOCKED", font=small, fill="white", anchor="mm")

    if not view.controls:
        draw.text((480, 610), "No controls in this room", font=label, fill="#60717c", anchor="mm")
    total = len(view.controls) * 340 + max(0, len(view.controls) - 1) * 28
    for index, control in enumerate(view.controls):
        x = (960 - total) / 2 + index * 368
        draw.rounded_rectangle((x, 540, x + 340, 686), radius=16, fill="white", outline="#b8c6cf", width=2)
        draw_symbol(draw, control.symbol, (x + 66, 612), 32)
        draw.text((x + 218, 572), control.id, font=label, fill=INK, anchor="mm")
        draw.rounded_rectangle((x + 172, 600, x + 264, 630), radius=15,
                               fill="#17654e" if control.is_on else "#778892")
        knob_x = x + (249 if control.is_on else 187)
        draw.ellipse((knob_x - 11, 604, knob_x + 11, 626), fill="white")
        draw.text((x + 218, 657), "ON" if control.is_on else "OFF", font=small, fill=INK, anchor="mm")
    return image


def render_maze(view: MazeView) -> Image.Image:
    """One stable floor plan. State is revealed only at the current room.

    A passage has two room-specific symbols and one shared gate. Its gate state
    is visible if either endpoint is current; all other gates remain unknown.
    """
    size, gap, margin, top = ROOM_SIZE, ROOM_GAP, MARGIN, TOP
    min_x = min(r.position[0] for r in view.rooms)
    min_y = min(r.position[1] for r in view.rooms)
    columns = max(r.position[0] for r in view.rooms) - min_x + 1
    rows = max(r.position[1] for r in view.rooms) - min_y + 1
    # Narrow generated floor plans still need room for the header and legend.
    width = 2 * margin + max(columns, 3) * size + (max(columns, 3) - 1) * gap
    height = top + rows * size + (rows - 1) * gap + 94
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    title = ImageFont.load_default(size=30)
    label = ImageFont.load_default(size=19)
    small = ImageFont.load_default(size=16)
    blue, green, red, muted = "#1565bb", "#17654e", "#a33f31", "#697a87"
    wall, floor, active_floor = "#465b6a", "#ffffff", "#e5f1ff"
    draw.text((margin, 28), "MAZE", fill=INK, font=title)
    draw.text((margin, 70), "Find your blue marker. Door and switch states are visible only in your room.",
              fill=muted, font=label)

    def center(position):
        return room_center(view, position)

    def badge(x, y, text, color):
        w = max(36, draw.textlength(text, font=small) + 16)
        draw.rounded_rectangle((x - w / 2, y - 13, x + w / 2, y + 13), radius=5, fill=color)
        draw.text((x, y), text, font=small, fill="white", anchor="mm")

    by_position = {r.position: r for r in view.rooms}
    passages = []
    for room in view.rooms:
        for door in room.doors:
            if room.position >= door.neighbor:
                continue
            other = by_position[door.neighbor]
            reverse = next(d for d in other.doors if d.neighbor == room.position)
            state = door.is_open if room.current else reverse.is_open if other.current else None
            passages.append((center(room.position), center(door.neighbor), state))

    # Corridors are part of the floor plan, underneath the room floors/walls.
    for (x1, y1), (x2, y2), _ in passages:
        draw.line((x1, y1, x2, y2), fill=wall, width=60)
        draw.line((x1, y1, x2, y2), fill=floor, width=50)
    for room in view.rooms:
        x, y = center(room.position)
        fill = active_floor if room.current else floor
        outline = blue if room.current else wall
        box = (x - size / 2, y - size / 2, x + size / 2, y + size / 2)
        draw.rectangle(box, fill=fill, outline=outline, width=6)
        for door in room.doors:
            dx = door.neighbor[0] - room.position[0]
            dy = door.neighbor[1] - room.position[1]
            wx, wy = x + dx * size / 2, y + dy * size / 2
            if dx:
                draw.rectangle((wx - 6, wy - 24, wx + 6, wy + 24), fill=fill)
            else:
                draw.rectangle((wx - 24, wy - 6, wx + 24, wy + 6), fill=fill)
            sx, sy = x + dx * 104, y + dy * 104
            draw_symbol(draw, door.symbol, (sx, sy), 16, fill)
            if door.id is not None:
                tx, ty = (sx, sy + 33) if dx else (sx + 64, sy)
                draw.text((tx, ty), door.id, font=small, fill=blue, anchor="mm")
        draw_symbol(draw, room.emblem, (x, y), 30, fill)
        if room.current:
            draw.ellipse((x - 12, y + 42, x + 12, y + 66), fill=blue, outline="white", width=3)
            draw.text((x, y + 83), "YOU", font=small, fill=blue, anchor="mm")
        for index, control in enumerate(room.controls):
            cx, cy = x + (-80 if index == 0 else 80), y - 70
            draw_symbol(draw, control.symbol, (cx, cy), 16, fill)
            draw.text((cx, cy + 29), control.id or "switch", font=small, fill=INK, anchor="mm")
            text = "?" if control.is_on is None else "ON" if control.is_on else "OFF"
            color = muted if control.is_on is None else green if control.is_on else red
            badge(cx, cy - 31, text, color)

    # Gates sit in the corridor; distant unknown gates are never colored by state.
    for (x1, y1), (x2, y2), state in passages:
        x, y = (x1 + x2) / 2, (y1 + y2) / 2
        color = muted if state is None else green if state else red
        spans = ((-24, -13), (13, 24)) if state is True else ((-24, 24),)
        for a, b in spans:
            line = (x, y + a, x, y + b) if y1 == y2 else (x + a, y, x + b, y)
            draw.line(line, fill=color, width=8)
        if state is None:
            draw.ellipse((x - 11, y - 11, x + 11, y + 11), fill=muted)
            draw.text((x, y), "?", font=small, fill="white", anchor="mm")
        else:
            bx, by = (x, y - 47) if y1 == y2 else (x + 74, y)
            badge(bx, by, "OPEN" if state else "LOCKED", color)

    footer_y = height - 49
    badge(margin + 30, footer_y, "YOU", blue)
    draw.text((margin + 76, footer_y), "Current room", font=label, fill=INK, anchor="lm")
    badge(margin + 273, footer_y, "?", muted)
    draw.text((margin + 309, footer_y), "State hidden until you enter the room", font=label, fill=INK, anchor="lm")
    draw.text((width - margin, footer_y), "Only local IDs can be used for actions", font=small, fill=muted, anchor="rm")
    return image
