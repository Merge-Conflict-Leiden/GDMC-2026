"""
This module defines various utility functions used throughout the project.
"""

import logging

from gdpc.interface import runCommand

logger = logging.getLogger(__name__)


def rotate_offset(sx: int, sz: int, W: int, D: int, direction: int) -> tuple[int, int]:
    """
    Map schematic cell (sx, sz) to world offset (ox, oz) from anchor.

    Verified empirically against GDPC rotatedBoxTransform (Box((0,0,0),(W,H,D))):
      d=0: identity
      d=1: 90° CW  — (sx,sz) → (W-1-sz, sx)
      d=2: 180°    — (sx,sz) → (W-1-sx, D-1-sz)
      d=3: 270° CW — (sx,sz) → (sz,     D-1-sx)

    Pass W=1, D=1 to rotate a plain displacement vector around the origin.
    """
    if direction == 1:
        return W - 1 - sz, sx
    if direction == 2:
        return W - 1 - sx, D - 1 - sz
    if direction == 3:
        return sz, D - 1 - sx
    return sx, sz


def seated_path_height(
    feat,
    path: list[tuple[int, int]],
    i: int,
    depth: int,
    width: int,
    max_cut: int = 2,
) -> int:
    """
    Surface Y for approach-path cell ``path[i]``, seated toward the LOWER
    terrain flanking it (perpendicular to the direction of travel).

    A path clamped only to its own cell's height rides the HIGH shoulder
    whenever it runs along a slope edge, ending up propped above the ground
    beside it.  Taking the lower flank sinks the track into the terrain
    instead.  Water flanks are ignored (a bank-side path must not drop to the
    riverbed) and the cut into the cell's own ground is capped at *max_cut*
    so a path along a cliff edge never digs a trench.
    """
    plz, plx = path[i]
    own = int(feat[plz, plx]["height"]) - 1
    if i + 1 < len(path):
        dz, dx = path[i + 1][0] - plz, path[i + 1][1] - plx
    elif i > 0:
        dz, dx = plz - path[i - 1][0], plx - path[i - 1][1]
    else:
        return own

    lo = own
    for fz, fx in ((plz + dx, plx + dz), (plz - dx, plx - dz)):  # perpendicular
        if (
            0 <= fz < depth
            and 0 <= fx < width
            and float(feat[fz, fx]["water_pct"]) == 0.0
        ):
            lo = min(lo, int(feat[fz, fx]["height"]) - 1)
    return max(lo, own - max_cut)


def sign_nbt(lines: list[str], limit: int = 15) -> str:
    """
    Build sign block-entity NBT (MC 1.21.5 component format) for up to four
    front-text lines.  Characters that would break the SNBT string are stripped
    and each line is clamped to *limit* characters so it fits on a sign face.
    """

    def _san(text: str) -> str:
        return text.replace("\\", "").replace('"', "").replace("'", "")[:limit]

    msgs = [_san(line) for line in lines[:4]]
    while len(msgs) < 4:
        msgs.append("")
    body = ",".join(f'"{m}"' for m in msgs)
    return f"{{front_text:{{messages:[{body}]}}}}"


def setup_logging(verbosity: int = 2) -> None:
    level = [logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG][
        min(verbosity, 3)
    ]

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=level,
            format="[%(levelname)s - %(filename)s, line %(lineno)d]: %(message)s",
        )
    else:
        root_logger.setLevel(level)


def log(message: str, in_game: bool = False, verbosity: int = logging.INFO):
    """
    Log a message in the console and (optionally) in-game through the chat interface.

    :param message: The message to log
    :param in_game: Whether to log the message in-game as well; default is False
                    so non-interactive submission runs do not pay an HTTP "say"
                    per log line.  Pass True when actively watching in-game.
    """

    match verbosity:
        case logging.DEBUG:
            logger.debug(message)
        case logging.WARNING:
            logger.warning(message)
        case logging.ERROR:
            logger.error(message)
        case _:
            logger.info(message)
    if in_game:
        runCommand(f"say {message}")


def announce(message: str) -> None:
    """
    Send a short, friendly status update to the in-game chat, in Dutch.

    Unlike ``log()``, this is player-facing narration rather than a technical
    trace: it is always shown in-game (no verbosity gate) and should only be
    called at major pipeline milestones, not per-item detail.

    :param message: The Dutch chat message to broadcast.
    """
    logger.info(message)
    runCommand(f"say {message}")
