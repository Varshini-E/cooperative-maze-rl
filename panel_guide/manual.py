"""Build the Expert's private manual without exposing state or physical slots."""

from panel_guide.tasks import MazeTask


def build_manual(task: MazeTask) -> str:
    lines = [
        "MAZE MANUAL",
        f"Goal: reach {task.goal} (room emblem: {task.room(task.goal).emblem}).",
        "Door symbols identify routes within a room; their meaning can differ elsewhere.",
        "A toggle reverses a switch value. Values persist when you leave a room.",
        "Initial location and switch values are unknown. Ask the Operator for observations.",
        "Door positions are not specified here. Match symbols to the visible local targets.",
    ]
    # Semantic ordering keeps physical slot order out of the manual.
    for room in sorted(task.rooms, key=lambda r: r.id):
        lines.extend(("", f"{room.id} (emblem: {room.emblem})"))
        for door in sorted(room.doors, key=lambda d: d.symbol):
            condition = " and ".join(f"{r.switch} {'on' if r.value else 'off'}" for r in door.requirements)
            rule = f"requires {condition}." if condition else "always open."
            lines.append(f"- {door.symbol} door -> {door.destination}; {rule}")
        for control in sorted(room.controls, key=lambda c: c.symbol):
            lines.append(f"- {control.symbol} control toggles switch {control.switch}.")
    return "\n".join(lines)
