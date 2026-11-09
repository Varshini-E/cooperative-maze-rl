"""Shared appearance names only; no task-specific state, routes or goal."""

CANONICAL_NAMES = {
    "solid circle": "solid_circle_control",
    "striped circle": "striped_circle_control",
}

APPEARANCES = {
    "ring": "hollow circular outline",
    "bars": "three parallel horizontal strokes",
    "solid circle": "filled circular control",
    "striped circle": "circular control containing vertical stripes",
    "plus": "horizontal and vertical strokes crossing at the center",
    "triangle": "filled triangle pointing upward",
    "diamond": "filled four-point diamond",
    "zigzag": "angular bent line with horizontal segments and diagonal joins",
    "crescent": "curved moon shape with an open cutout",
    "sun": "filled central disk with eight separate rays",
    "star": "filled five-point star",
    "leaf": "pointed leaf with a stem and visible veins",
    "anvil": "wide flat top above a narrow waist and flared base",
}


def canonical_name(symbol):
    return CANONICAL_NAMES.get(symbol, symbol)


SHARED_LEGEND = "\nSHARED VISUAL LEGEND\n" + "\n".join(
    f"{canonical_name(symbol)}: {appearance}." for symbol, appearance in APPEARANCES.items()
) + """
Use these canonical names when the appearance is identifiable; otherwise say
uncertain and describe what you can see. A bare 'circle' is ambiguous: do not
assume ring, a filled control, or a striped control. 'Three horizontal lines'
describes bars. 'Solid circle control' and 'striped circle control' are descriptive
synonyms for the two control names. Door symbols and control symbols are different
object roles; identify the role and current local ID along with the symbol.
The legend supplies terminology, not facts about which objects are in this maze.
"""

STRICT_EXPERT_CONTRACT = """
STRICT EXPERT MESSAGE CONTRACT
Every final public Expert response must terminate cleanly and be exactly one JSON object with
type="ask" or type="instruct" and a nonempty text string, with no other fields.
The first invalid Expert response ends the episode with reward 0; no Operator
call or physical action occurs on that exchange. There is no correction attempt.
Valid JSON alone earns nothing: the pair must reach the goal within both budgets.
"""
