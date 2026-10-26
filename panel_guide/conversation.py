"""The Expert's environment: a frozen partner, protocol, and physical simulator."""

import asyncio
import io
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from panel_guide.engine import Action, MazeGame, RoomView
from panel_guide.manual import build_manual
from panel_guide.protocol import Message, ProtocolError, operator_response_schema, parse_message
from panel_guide.render import crop_current_room, render_maze, render_room, render_version
from panel_guide.tasks import MazeTask

EXPERT_PROMPT = """You are the text Expert guiding a partner through a maze.
You have the private manual but no images or current hidden state. Ask your partner
for relevant observations, track location and switches, and give unambiguous local
instructions. You may communicate any manual information. Reach the manual's goal
within both budgets. A move or toggle must be performed by the Operator.
Return exactly one JSON object, no Markdown or private reasoning:
{"type":"ask","text":"your question"} or
{"type":"instruct","text":"your instruction"}.
An ask cannot cause an action. An instruct permits at most one physical action.
The user message contains the shared transcript and remaining budgets.
"""

OPERATOR_PROMPT = """You are the visual Operator in a cooperative maze.
Use only the current screenshot and shared messages. You have no private map.
Report visible facts accurately; use unclear when an instruction is ambiguous.
Your latest Expert message is in the shared transcript. For ask, respond with text
only. For instruct, respond with text or perform at most one action using a local
target ID visible in the screenshot. Do not invent observations or target IDs.
Return exactly one JSON object, no Markdown or private reasoning:
{"type":"reply","text":"your observations"},
{"type":"unclear","text":"your clarification question"},
{"type":"move","target":"door_1"}, or
{"type":"toggle","target":"control_1"}.
Prior screenshots are unavailable; the shared text is your persistent memory.
"""

OPERATOR_PROMPT_V2 = OPERATOR_PROMPT.replace(
    "only. For instruct, respond with text or perform at most one action using a local\n"
    "target ID visible in the screenshot. Do not invent observations or target IDs.",
    "only. For instruct, execute exactly one grounded move or toggle if possible,\n"
    "using a local target ID visible in the screenshot. If the action is ambiguous,\n"
    "unavailable, locked, or unnecessary, return unclear with the specific reason\n"
    "no action can or should be executed. A generic description is not an answer\n"
    "to an instruct message. Never substitute a description for an executable action.\n"
    "Do not invent observations or target IDs.",
)

EXPERT_PROMPT_V2 = EXPERT_PROMPT + """Use confirmed execution outcomes to track progress.
If no action was executed, the physical situation has not changed. After a move,
old door IDs and observations refer to the previous room. Obtain current visible
details when needed before giving another local instruction.
"""

OPERATOR_PROMPT_V3 = OPERATOR_PROMPT_V2 + """The final user message includes the response
schema for this turn. A move or toggle is a tool call, not confirmation of success.
Only the controller's execution result confirms whether it took effect. Responses
outside the current schema are rejected without execution. Return one object only.
"""

EXPERT_MAZE_VIEW = """
Your visual partner sees one complete floor plan: all room emblems, passages, door
symbols, and control symbols. A blue marker identifies its current room. Only
that room's door/switch states and local action IDs are revealed. Remote states
are marked unknown, even after a visit; shared text can retain previous reports.
The private goal, switch identities, and gate rules are not drawn on the map.
"""

OPERATOR_MAZE_VIEW = """
The screenshot is ONE complete maze floor plan. The blue room and YOU marker
identify your current location. Large central symbols are room emblems; smaller
symbols beside wall openings identify doors FROM THAT SIDE of the passage.
Unless asked about the wider map, report the current room's details.
Only the current room has action IDs. A move uses a door_N in that room; a toggle
uses a control_N in that room. Other rooms are visible but cannot be acted on
remotely. Each corridor has one gate: OPEN/LOCKED is shown only for passages
touching the current room. A gray ? means UNKNOWN, not open, closed, or OFF.
Other rooms' control states are also unknown. Previously visited rooms do not
retain state indicators; use shared text for remembered observations. The map
does not disclose the goal or switch rules. An ID's number is not a compass direction.
"""

EXPERT_PROMPT_V4 = EXPERT_PROMPT_V2 + """
Only YOU have the private manual. Apply its room mapping, goal, and switch rules
yourself; do not ask the Operator to consult it. You may forward relevant rules
or delegate a decision by explicitly providing the information it needs.
Before choosing a route, establish the CURRENT room emblem and each local door's
symbol, target ID, and OPEN/LOCKED state. Ask for local control symbols and ON/OFF
states when relevant. 'The blue room' or door IDs alone do not identify a room.
Match the reported emblem and symbols to YOUR manual, then instruct one concrete
action. Use ask for inspection; use instruct for execution. A toggle reverses the
switch: preserve values that already satisfy the goal's rules.
Track confirmed moves, toggles, and observations in the shared text. A satisfied
response means the Operator reports your requested condition already holds; no
action occurred and the controller has not independently verified that claim.
"""

OPERATOR_PROMPT_V4 = """You are the frozen visual Operator in a cooperative maze.
You can inspect current images and execute one LOCAL action. Only the Expert has
the private manual and goal; use any rules the Expert explicitly communicates.
Answer the CURRENT Expert message. Ground facts in the current blue room and YOU
marker, even when older descriptions disagree. Never invent local IDs or states.
For a general room report, include the central ROOM EMBLEM, every LOCAL door's
ID + symbol + OPEN/LOCKED state, and every LOCAL control's ID + symbol + ON/OFF
state, or 'controls: none'. Keep it compact. If asked only a specific fact, answer
that fact. Door symbols near openings are not room emblems. Remote objects are
not local targets. A gray ? means unknown, never OFF, OPEN, or LOCKED.
Return exactly one JSON object with no Markdown or private reasoning:
For ask: {"type":"reply","text":"observations"} or
{"type":"unclear","text":"specific missing or ambiguous information"}.
For instruct: {"type":"move","target":"door_N"} or
{"type":"toggle","target":"control_N"} for one grounded executable action;
{"type":"satisfied","text":"which requested condition already holds"} when
the instruction is already satisfied, e.g. ensuring an already-ON control is ON;
or {"type":"unclear","text":"specific reason execution is impossible"} for a
locked door, absent local target, ambiguous instruction, or missing information.
Do not move through a visibly LOCKED door. Do not replace an executable action
with a room description. A tool call requests an operation; only the controller's
feedback confirms execution. The final user message gives this turn's schema.
"""

# v5 changes only the Operator's primitive-action contract; Expert v4 is unchanged.
OPERATOR_PROMPT_V5 = OPERATOR_PROMPT_V4 + """
TOGGLE means reverse the current switch value exactly once: OFF becomes ON and
ON becomes OFF. An unconditional 'toggle' is never already satisfied. Return a
toggle tool call for the named local control in either state.
'Ensure ON' is different: toggle if OFF, otherwise report satisfied.
'Ensure OFF' means toggle if ON, otherwise report satisfied.
Use satisfied only for an explicitly requested state that already holds, never
as a substitute for an unconditional toggle or move.
"""

# v6 tolerates inspection mislabeled as instruct; actions still require tool calls.
# Keep the Expert, toggle contract, images, and history policy fixed for comparison.
OPERATOR_PROMPT_V6 = OPERATOR_PROMPT_V5.replace(
    'For instruct: {"type":"move","target":"door_N"} or',
    'For instruct: if the Expert requests an observation, answer with\n'
    '{"type":"reply","text":"observations"} even though the turn is labeled instruct.\n'
    'For an executable action, return {"type":"move","target":"door_N"} or',
)

# Prompt correction is independent of operator_history, so it can be compared
# with v6 under the same latest-only context before testing recent text memory.
OPERATOR_PROMPT_V7 = """You are the visual Operator helping a text Expert solve a maze.
Use the current images to answer the Expert or request one local physical action.
Only the Expert has the private manual and goal; use rules it explicitly shares.

For ask, answer every requested observable item in ONE reply. Identify anything
you cannot determine instead of guessing. Related questions need no extra call
or separate observation phase. For a general room report, include the current
room emblem, every local door's ID, symbol and OPEN/LOCKED state, and each local
control's ID, symbol and ON/OFF state, or say there are no local controls.
For instruct, request one grounded move or toggle, or explain specifically why
you cannot execute it. If a question is mislabeled instruct, answer it with reply.
Do not replace an executable action with a description.

Current images determine current visual state. Previous Operator reports are
unverified and can be mistaken or outdated. Local IDs belong to the room where
they were observed; after movement, ground them again in the current image.
Only a controller receipt with executed=true confirms that an action took effect;
an earlier move/toggle response is just a request. Never infer success from prose.
Use the blue room and YOU marker. Remote objects are not local targets. A gray ?
means unknown. Do not move through a visibly LOCKED door or invent target IDs.

Toggle reverses a switch exactly once, ON to OFF or OFF to ON. An unconditional
toggle is never already satisfied. Ensure ON means toggle if OFF, otherwise
satisfied; ensure OFF means toggle if ON, otherwise satisfied. Use satisfied
only when an explicitly requested state already holds, never for an unconditional
move/toggle or a question asking you to inspect a state.

Return exactly one JSON object, with no Markdown or private reasoning. The final
message supplies the permitted response schema. These examples illustrate the
format only; their room facts and IDs are NOT facts about your current images.
Read your images and supply your own answer; never copy an example as evidence.
Complete factual reply:
{"type":"reply","text":"Room emblem: leaf. door_1: triangle, OPEN. door_2: plus, LOCKED. control_1: solid circle, ON."}
Reply with explicit uncertainty:
{"type":"reply","text":"door_2 is LOCKED. Its symbol is too small for me to identify in the current images."}
Action request:
{"type":"move","target":"door_1"}
{"type":"toggle","target":"control_1"}
Specific execution problem:
{"type":"unclear","text":"I cannot move through door_2 because it is visibly LOCKED."}
Already-met explicit condition:
{"type":"satisfied","text":"You asked to ensure control_1 is ON; it is already ON, so no toggle is needed."}
"""


class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class Request:
    role: str
    system: str
    history: tuple[dict, ...]
    remaining_exchanges: int
    remaining_actions: int
    image_png: bytes | None = None
    # Only the free scripted test partner uses these already-visible fields.
    # The Tinker adapter forwards image_png, never this structured observation.
    scripted_view: RoomView | None = None
    focus_current: bool = False
    operator_turn: str | None = None
    observation_mode: str = "room"
    detail_png: bytes | None = None
    protocol_version: str = "v3"
    operator_history: str = "full"

    def text_messages(self) -> list[dict]:
        messages = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": json.dumps({
                "shared_transcript": self.history,
                "remaining_exchanges_after_this_one": self.remaining_exchanges,
                "remaining_actions": self.remaining_actions,
            })},
        ]
        if self.focus_current:
            if self.role == "operator":
                current = next((event for event in reversed(self.history) if event["role"] == "expert"), None)
                text = "Respond to this CURRENT EXPERT MESSAGE. Earlier messages are history.\n"
                text += json.dumps(current)
            else:
                latest_operator = next((event for event in reversed(self.history) if event["role"] == "operator"), None)
                latest_feedback = next((event for event in reversed(self.history) if event["role"] == "environment"), None)
                text = "Choose your next question or instruction using the latest confirmed information.\n"
                text += json.dumps({"latest_operator_message": latest_operator, "latest_execution_feedback": latest_feedback})
            messages.append({"role": "user", "content": text})
        if self.operator_turn is not None:
            if self.role != "operator" or not self.focus_current:
                raise ValueError("A turn contract requires a focused Operator request.")
            messages[-1]["content"] += "\nCURRENT TURN CONTRACT (JSON Schema):\n"
            messages[-1]["content"] += json.dumps(operator_response_schema(self.operator_turn, self.protocol_version))
        return messages


@dataclass(frozen=True)
class Completion:
    text: str
    stop_reason: str = "stop"
    sample: dict[str, Any] = field(default_factory=dict)


class Agent(Protocol):
    async def complete(self, request: Request) -> Completion: ...


@dataclass(frozen=True)
class EpisodeResult:
    status: str
    reason: str
    reward: float
    exchanges: int
    actions: int


class Recorder:
    """Flush every event so an interrupted API run still leaves a usable trace."""

    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        if any(directory.iterdir()):
            raise ValueError("Output directory must be empty; keep previous runs for comparison.")
        self.directory = directory

    def event(self, **event: Any) -> None:
        with (self.directory / "trace.jsonl").open("a") as stream:
            stream.write(json.dumps(event) + "\n")

    def audit(self, **event: Any) -> None:
        with (self.directory / "audit.jsonl").open("a") as stream:
            stream.write(json.dumps(event) + "\n")

    def request(self, exchange: int, request: Request) -> None:
        image_path, detail_path = None, None
        if request.image_png is not None:
            prefix = "maze" if request.observation_mode.startswith("maze") else "room"
            image_path = f"{prefix}-{exchange:02d}.png"
            (self.directory / image_path).write_bytes(request.image_png)
        if request.detail_png is not None:
            detail_path = f"detail-{exchange:02d}.png"
            (self.directory / detail_path).write_bytes(request.detail_png)
        self.event(type="request", exchange=exchange, role=request.role,
                   messages=request.text_messages(), image=image_path, detail_image=detail_path,
                   protocol_version=request.protocol_version, operator_history=request.operator_history,
                   observation_mode=request.observation_mode,
                   renderer=render_version(request.observation_mode) if image_path else None)

    def finish(self, result: EpisodeResult) -> None:
        self.event(type="terminal", **asdict(result))
        (self.directory / "summary.json").write_text(json.dumps(asdict(result), indent=2) + "\n")


class Conversation:
    def __init__(self, task: MazeTask, *, max_actions: int = 16, max_exchanges: int = 32,
                 prompt_version: str = "v1", expert_instructions: str = "",
                 observation_mode: str = "maze", operator_history: str = "full") -> None:
        if type(max_exchanges) is not int or max_exchanges <= 0:
            raise ValueError("max_exchanges must be a positive integer.")
        self.task = task  # Controller/audit only; never included in a model Request.
        self.game = MazeGame(task, max_actions)
        if prompt_version not in ("v1", "v2", "v3", "v4", "v5", "v6", "v7"):
            raise ValueError("prompt_version must be v1, v2, v3, v4, v5, v6, or v7.")
        self.prompt_version = prompt_version
        self.protocol_version = "v6" if prompt_version in ("v6", "v7") else "v4" if prompt_version in ("v4", "v5") else "v3"
        if operator_history not in ("full", "latest", "recent2"):
            raise ValueError("operator_history must be full, latest, or recent2.")
        self.operator_history = operator_history
        render_version(observation_mode)  # Validate before any model calls.
        self.observation_mode = observation_mode
        self.expert_instructions = expert_instructions
        self.manual = build_manual(task)
        self.max_actions = max_actions
        self.max_exchanges = max_exchanges
        self.exchanges = 0
        self.history: list[dict] = []
        self.finished = False
        self.result: EpisodeResult | None = None
        self.recorder: Recorder | None = None
        self._started = False
        self._awaiting_expert = False
        self._processing = False

    def _request(self, role: str) -> Request:
        png, detail_png, view, operator_turn = None, None, None, None
        expert_prompt = EXPERT_PROMPT if self.prompt_version == "v1" else EXPERT_PROMPT_V4 if self.prompt_version in ("v4", "v5", "v6", "v7") else EXPERT_PROMPT_V2
        system = expert_prompt + "\nPRIVATE MANUAL\n" + self.manual
        if self.observation_mode in ("maze", "maze_detail"):
            system += EXPERT_MAZE_VIEW
        if self.operator_history == "latest":
            system += "\nThe Operator receives only your latest message and current images. Keep navigation memory yourself; repeat any rule/context needed for the current instruction.\n"
        elif self.operator_history == "recent2":
            system += "\nThe Operator receives your current message and the last two completed exchanges, including controller feedback, with current images only. Keep the longer navigation history yourself. Make requests clear and repeat any older rule/context needed now.\n"
        system += self.expert_instructions
        if role == "operator":
            system = {"v1": OPERATOR_PROMPT, "v2": OPERATOR_PROMPT_V2,
                      "v3": OPERATOR_PROMPT_V3, "v4": OPERATOR_PROMPT_V4,
                      "v5": OPERATOR_PROMPT_V5, "v6": OPERATOR_PROMPT_V6,
                      "v7": OPERATOR_PROMPT_V7}[self.prompt_version]
            if self.observation_mode in ("maze", "maze_detail"):
                system = system.replace("You have no private map.", "You do not have the private manual.")
                system += OPERATOR_MAZE_VIEW
            if self.observation_mode == "maze_detail":
                system += "\nImage 1 is the whole maze. Image 2 is an enlarged crop of the CURRENT room and its adjoining passages from that same image. Use the crop for local labels and states; neighboring room edges are not local objects. No hidden state is added.\n"
            if self.operator_history == "latest":
                system = system.replace("Prior screenshots are unavailable; the shared text is your persistent memory.", "")
                system += "\nYou receive only the latest Expert message and current images. The Expert retains navigation memory. Do not assume earlier room details.\n"
            elif self.operator_history == "recent2":
                system = system.replace("Prior screenshots are unavailable; the shared text is your persistent memory.", "")
                system += "\nYou receive the last two completed text exchanges with their controller feedback, followed by the current Expert message. Earlier reports are unverified; only controller receipts confirm execution. The attached images are current, not historical. Re-ground local IDs after movement and use current images for current visual state. No earlier conversation is available.\n"
            if self.prompt_version in ("v3", "v4", "v5", "v6", "v7"):
                operator_turn = next(event["type"] for event in reversed(self.history) if event["role"] == "expert")
            view = self.game.view
            buffer = io.BytesIO()
            image = render_maze(self.game.maze_view) if self.observation_mode in ("maze", "maze_detail") else render_room(view)
            with image:
                image.save(buffer, format="PNG")
                if self.observation_mode == "maze_detail":
                    detail_buffer = io.BytesIO()
                    with crop_current_room(image, self.game.maze_view) as detail:
                        detail.save(detail_buffer, format="PNG")
                    detail_png = detail_buffer.getvalue()
            png = buffer.getvalue()
        # Copy the history: a partner must not mutate the controller's transcript.
        history = tuple(json.loads(json.dumps(self.history)))
        if role == "operator" and self.operator_history == "latest":
            history = tuple(e for e in history if e["role"] == "expert")[-1:]
        elif role == "operator" and self.operator_history == "recent2":
            # Feedback ends an exchange, including rejected-output exchanges.
            # Keep the final two completed exchanges plus the current message,
            # rather than an arbitrary number of messages that could split a receipt.
            endings = [i for i, event in enumerate(history) if event["role"] == "environment"]
            history = history[endings[-3] + 1:] if len(endings) > 2 else history
        return Request(role=role, system=system, history=history,
                       remaining_exchanges=self.max_exchanges - self.exchanges,
                       remaining_actions=self.game.remaining_actions, image_png=png, scripted_view=view,
                       focus_current=self.prompt_version != "v1", operator_turn=operator_turn,
                       observation_mode=self.observation_mode, detail_png=detail_png,
                       protocol_version=self.protocol_version, operator_history=self.operator_history)

    def _feedback(self, recorder: Recorder | None, **fields: Any) -> None:
        if self.prompt_version in ("v3", "v4", "v5", "v6", "v7"):
            fields.setdefault("outcome", "no_action")
            # executed means the physical operation took effect; a blocked call
            # still consumes an action attempt and has a correlated tool result.
            fields["executed"] = fields["outcome"] in ("moved", "toggled")
            if not fields["executed"]:
                fields["detail"] = (fields.get("detail", "") +
                                    " No action was executed; the physical state is unchanged.").strip()
        event = {"role": "environment", "remaining_actions": self.game.remaining_actions,
                 "remaining_exchanges": self.max_exchanges - self.exchanges, **fields}
        self.history.append(event)
        if recorder:
            recorder.event(type="feedback", exchange=self.exchanges, **event)

    def start(self, recorder: Recorder | None = None) -> Request:
        """Begin a single-use episode and expose its first Expert observation."""
        if self._started or self.finished or self.exchanges:
            raise RuntimeError("Create a new Conversation for each episode.")
        self._started = True
        self.recorder = recorder
        if recorder:
            recorder.audit(type="initial_state", state=asdict(self.game.audit_state))
            (recorder.directory / "manual.txt").write_text(self.manual)
        return self.next_expert_request()

    def next_expert_request(self) -> Request:
        if not self._started or self.finished or self._awaiting_expert or self._processing:
            raise RuntimeError("An Expert observation must follow exactly one completed exchange.")
        self.exchanges += 1
        self._awaiting_expert = True
        request = self._request("expert")
        if self.recorder:
            self.recorder.request(self.exchanges, request)
        return request

    def _finish(self, status: str, reason: str, reward: float = 0.0) -> EpisodeResult:
        self.finished = True
        self._awaiting_expert = False
        self._processing = False
        self.result = EpisodeResult(status, reason, reward, self.exchanges,
                                    self.max_actions - self.game.remaining_actions)
        if self.recorder:
            self.recorder.finish(self.result)
        return self.result

    def interrupt(self, role: str, error: Exception) -> EpisodeResult:
        if not self._started or self.finished:
            raise RuntimeError("Only an active episode can be interrupted.")
        if not isinstance(error, BudgetExceeded) and self.recorder:
            # Exception text can contain credentials; persist only its class.
            self.recorder.event(type="provider_error", exchange=self.exchanges,
                                role=role, error_class=type(error).__name__)
        return self._finish("interrupted", "spending_limit" if isinstance(error, BudgetExceeded) else "infrastructure_error")

    def _accept(self, completion: Completion, role: str, *, expert_turn: str | None = None,
                latency_seconds: float = 0.0) -> Message | None:
        if self.recorder:
            self.recorder.event(type="completion", exchange=self.exchanges, role=role,
                                latency_seconds=latency_seconds, **asdict(completion))
        try:
            if completion.stop_reason != "stop":
                raise ProtocolError("Response was truncated or did not terminate cleanly.")
            operator_turn = expert_turn if self.prompt_version in ("v3", "v4", "v5", "v6", "v7") else None
            message = parse_message(completion.text, role, operator_turn=operator_turn,
                                    protocol_version=self.protocol_version)
            if expert_turn == "ask" and message.type in ("move", "toggle"):
                raise ProtocolError("An ask permits only reply or unclear.")
        except ProtocolError as exc:
            self._feedback(self.recorder, error=exc.code, sender=role, detail=str(exc))
            return None
        self.history.append({"role": role, **message.to_dict()})
        if self.recorder:
            self.recorder.event(type="accepted_message", exchange=self.exchanges,
                                role=role, message=message.to_dict())
        return message

    async def step(self, expert_completion: Completion, operator: Agent, *,
                   timeout_seconds: float = 120, expert_latency_seconds: float = 0.0) -> EpisodeResult | None:
        """Apply one Expert message, call the frozen partner, and advance the maze.

        None means the episode continues. Both the legacy runner and Cookbook
        Env use this same execution path, including invalid-output handling.
        """
        if not self._awaiting_expert or self.finished:
            raise RuntimeError("Start an exchange before supplying its Expert action.")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        self._awaiting_expert = False
        self._processing = True
        expert_message = self._accept(expert_completion, "expert", latency_seconds=expert_latency_seconds)
        if expert_message is not None:
            request = self._request("operator")
            if self.recorder:
                self.recorder.request(self.exchanges, request)
            started = time.monotonic()
            try:
                completion = await asyncio.wait_for(operator.complete(request), timeout_seconds)
            except Exception as exc:
                return self.interrupt("operator", exc)
            message = self._accept(completion, "operator", expert_turn=expert_message.type,
                                   latency_seconds=time.monotonic() - started)
            if message is not None:
                if message.type in ("move", "toggle"):
                    before, receipt = self.game.audit_state, {}
                    if self.prompt_version in ("v3", "v4", "v5", "v6", "v7"):
                        receipt = {"tool_call_id": f"exchange-{self.exchanges}"}
                        if self.recorder:
                            self.recorder.event(type="tool_call", exchange=self.exchanges, **receipt,
                                                name=message.type, arguments={"target": message.value})
                    result = self.game.step(Action(message.type, message.value))
                    self._feedback(self.recorder, **asdict(result), **receipt)
                    if self.recorder:
                        self.recorder.audit(type="transition", exchange=self.exchanges,
                                            before=asdict(before), after=asdict(self.game.audit_state))
                    if result.done:
                        return self._finish(result.status, result.termination_reason, result.reward)
                else:
                    fields = {"outcome": "no_action", "reward": 0.0}
                    if message.type == "satisfied":
                        fields.update(outcome="reported_satisfied", condition_verified=False,
                                      detail="The Operator reports the requested condition already holds; this claim is not independently verified.")
                    if self.prompt_version in ("v6", "v7") and message.type == "reply":
                        fields.update(observation_verified=False,
                                      detail="The Operator's report is not independently verified.")
                    if self.prompt_version == "v2":
                        fields["detail"] = "No action was executed; the physical state is unchanged."
                        if expert_message.type == "instruct" and message.type == "reply":
                            fields["error"] = "instruction_following_failure"
                            fields["detail"] += " An instruct turn requires move/toggle, or unclear with a specific reason."
                    self._feedback(self.recorder, **fields)
        if self.exchanges >= self.max_exchanges:
            return self._finish("failure", "exchange_limit")
        self._processing = False
        return None

    async def run(self, expert: Agent, operator: Agent, recorder: Recorder | None = None,
                  *, timeout_seconds: float = 120) -> EpisodeResult:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        request = self.start(recorder)
        while True:
            started = time.monotonic()
            try:
                completion = await asyncio.wait_for(expert.complete(request), timeout_seconds)
            except Exception as exc:
                return self.interrupt("expert", exc)
            result = await self.step(completion, operator, timeout_seconds=timeout_seconds,
                                     expert_latency_seconds=time.monotonic() - started)
            if result is not None:
                return result
            request = self.next_expert_request()
