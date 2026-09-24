"""String the rendered shots of an ad into one frame sequence, with the edit transitions between them.

Transitions (chosen by adup.ugc.director, config ugc.director.transitions): hard cut, dissolve, whip pan, dip to black
or white. Shot boundaries in the output are exact, so the pair can be split per shot afterwards.
"""

import cv2
import numpy as np


def transition_effect(f, kind, p, direction):
    """p in (0, 1]: strength of the effect at this frame (1 = at the cut)."""
    x = f.astype(np.float32)
    if kind == "dip_black":
        x *= 1 - p
    elif kind == "dip_white":
        x = x * (1 - p) + 255 * p
    elif kind == "whip":
        k = max(int(p * f.shape[1] * 0.12) // 2 * 2 + 1, 3)
        x = cv2.filter2D(x, -1, np.full((1, k), 1 / k, np.float32))
        x = np.roll(x, int(direction * p * f.shape[1] * 0.08), axis=1)
    return np.clip(x, 0, 255).astype(np.uint8)


def sequence_frames(shots, transitions, renderers, rng):
    """Yield (frame, camera velocity, camera_captured) for the whole ad with transitions applied between shots."""
    direction = rng.choice([-1, 1])
    held = []                                     # dissolve: tail of the previous shot, blended into this shot's head
    for i, (shot, ren) in enumerate(zip(shots, renderers)):
        t_in = transitions[i - 1] if i > 0 else None
        t_out = transitions[i] if i < len(shots) - 1 else None
        n = shot["frames"]
        prev_tail, held = held, []
        vel = ren.velocity()
        for j, f in enumerate(ren.frames()):
            if t_out and t_out["type"] == "dissolve" and j >= n - t_out["frames"]:
                held.append(f)
                continue
            if t_out and t_out["type"] in ("dip_black", "dip_white", "whip"):
                k = t_out["frames"] // 2
                if j >= n - k:
                    f = transition_effect(f, t_out["type"], (j - (n - k) + 1) / (k + 1), direction)
            if t_in and t_in["type"] == "dissolve" and j < len(prev_tail):
                a = (j + 1) / (len(prev_tail) + 1)
                f = (prev_tail[j].astype(np.float32) * (1 - a) + f.astype(np.float32) * a + 0.5).astype(np.uint8)
            if t_in and t_in["type"] in ("dip_black", "dip_white", "whip"):
                k = t_in["frames"] - t_in["frames"] // 2
                if j < k:
                    f = transition_effect(f, t_in["type"], 1 - j / (k + 1), -direction)
            yield f, vel[min(j, len(vel) - 1)], ren.camera_captured


def shot_ranges(shots, transitions):
    """Output frame range [start, end) of every shot; a dissolve's blended frames count as the head of the next shot."""
    ranges, pos = [], 0
    for i, s in enumerate(shots):
        n = s["frames"] - (transitions[i]["frames"] if i < len(shots) - 1 and transitions[i]["type"] == "dissolve" else 0)
        ranges.append((pos, pos + n))
        pos += n
    return ranges
