"""Write the difftree logo: a soft depth-2 tree whose leaves are Gaussians.

Run:  python docs/make_logo.py
Edge widths are the soft routing probabilities, and each leaf is drawn as a
Gaussian whose height is its path weight mu_l(x). The heaviest leaf is orange.
"""
from math import exp
from pathlib import Path

NAVY, BLUE, MID, SKY, ORANGE = "#0b2545", "#1f5f99", "#2a7fc1", "#5aa9dd", "#e07a1f"
W = H = 240

s_right = [0.7, 0.4, 0.8]  # P(go right) at each internal node, BFS order
nodes = {0: (120, 46), 1: (66, 108), 2: (174, 108)}
leaf_x = [38, 94, 146, 202]
base_y, sigma, bell_half = 214, 7.5, 24

reach = {0: 1.0}
for i in range(3):
    reach[2 * i + 1] = reach[i] * (1 - s_right[i])
    reach[2 * i + 2] = reach[i] * s_right[i]
leaf_p = [reach[3 + k] for k in range(4)]
best = max(range(4), key=lambda k: leaf_p[k])
heights = [16 + 70 * p for p in leaf_p]


def bell(cx, height):
    pts = []
    for j in range(41):
        dx = -bell_half + 2 * bell_half * j / 40
        pts.append(f"{cx + dx:.1f},{base_y - height * exp(-0.5 * (dx / sigma) ** 2):.1f}")
    return "M" + " L".join(pts) + " Z"


def sigmoid(cx, cy, r):
    pts = []
    for j in range(21):
        u = -1 + 2 * j / 20
        pts.append(f"{cx + 0.62 * r * u:.1f},{cy + 0.45 * r * (1 - 2 / (1 + exp(-5 * u))):.1f}")
    return "M" + " L".join(pts)


parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
         "<title>difftree</title>"]

for parent in range(3):
    x0, y0 = nodes[parent]
    for child in (2 * parent + 1, 2 * parent + 2):
        if child in nodes:
            x1, y1 = nodes[child]
        else:
            k = child - 3
            x1, y1 = leaf_x[k], base_y - heights[k]
        width = 2.5 + 15 * reach[child]
        color = ORANGE if child == 3 + best or (parent == 0 and child == 2) else SKY
        parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1:.1f}" stroke="{color}" '
                     f'stroke-width="{width:.1f}" stroke-linecap="round" opacity="0.9"/>')

for k, cx in enumerate(leaf_x):
    fill = ORANGE if k == best else MID
    parts.append(f'<path d="{bell(cx, heights[k])}" fill="{fill}" fill-opacity="0.85" '
                 f'stroke="{NAVY}" stroke-width="2" stroke-linejoin="round"/>')
parts.append(f'<line x1="10" y1="{base_y}" x2="230" y2="{base_y}" stroke="{NAVY}" '
             f'stroke-width="3" stroke-linecap="round"/>')

for i, (cx, cy) in nodes.items():
    r = 17 if i == 0 else 14
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{NAVY}" stroke="{BLUE}" stroke-width="2"/>')
    parts.append(f'<path d="{sigmoid(cx, cy, r)}" fill="none" stroke="white" '
                 f'stroke-width="2.5" stroke-linecap="round"/>')

parts.append("</svg>")
out = Path(__file__).parent / "_static" / "logo.svg"
out.parent.mkdir(exist_ok=True)
out.write_text("\n".join(parts) + "\n")
print(f"wrote {out}")
