"""Write an SMIL-animated SVG of soft routing through a depth-3 tree.

Run:  python presentation/make_tree_svg.py
Edge widths grow with the probability of reaching each child, one depth at a
time, then the leaf weights mu_l(x) fill in. The cycle repeats every DUR s.
"""
from pathlib import Path

NAVY, BLUE, MID, SKY, ICE, ORANGE = "#0b2545", "#1f5f99", "#2a7fc1", "#5aa9dd", "#dbeaf7", "#e07a1f"
DUR = 8.0
W, H = 900, 470

# P(go right) at each internal node, BFS order
s_right = [0.75, 0.3, 0.9, 0.5, 0.2, 0.6, 0.95]
pos = {0: (450, 60), 1: (250, 150), 2: (650, 150),
       3: (150, 240), 4: (350, 240), 5: (550, 240), 6: (750, 240)}
leaf_x = [100 + 100 * i for i in range(8)]
leaf_y = 330

# reach probability for every node (internal 0..6, leaves 7..14)
reach = {0: 1.0}
for i in range(7):
    reach[2 * i + 1] = reach[i] * (1 - s_right[i])
    reach[2 * i + 2] = reach[i] * s_right[i]
for k, x in enumerate(leaf_x):
    pos[7 + k] = (x, leaf_y)


def depth(n):
    return 0 if n == 0 else (1 if n <= 2 else (2 if n <= 6 else 3))


def kt(t0, t1):
    """keyTimes for: idle until t0, grow until t1, hold, reset at end."""
    a, b = t0 / DUR, t1 / DUR
    return f"0;{a:.3f};{b:.3f};0.94;1"


parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
         f'font-family="Inter, Helvetica, Arial, sans-serif">']
parts.append(f'<rect width="{W}" height="{H}" fill="white"/>')

# edges
for parent in range(7):
    x0, y0 = pos[parent]
    for child, lab in ((2 * parent + 1, 1 - s_right[parent]), (2 * parent + 2, s_right[parent])):
        x1, y1 = pos[child]
        t0 = 0.6 + 1.2 * depth(parent)
        w = 1.5 + 16 * reach[child]
        parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" stroke="{ICE}" stroke-width="3"/>')
        parts.append(
            f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" stroke="{MID}" stroke-linecap="round" '
            f'stroke-width="0">'
            f'<animate attributeName="stroke-width" dur="{DUR}s" repeatCount="indefinite" '
            f'values="0;0;{w:.2f};{w:.2f};0" keyTimes="{kt(t0, t0 + 0.9)}"/></line>')
        # split-probability label on the edge
        mx, my = (x0 + x1) / 2 + (-14 if x1 < x0 else 14), (y0 + y1) / 2 - 4
        anchor = "end" if x1 < x0 else "start"
        parts.append(
            f'<text x="{mx:.0f}" y="{my:.0f}" font-size="15" fill="{BLUE}" text-anchor="{anchor}" opacity="0">'
            f'{lab:.2f}<animate attributeName="opacity" dur="{DUR}s" repeatCount="indefinite" '
            f'values="0;0;1;1;0" keyTimes="{kt(t0, t0 + 0.5)}"/></text>')

# internal nodes
for n in range(7):
    x, y = pos[n]
    parts.append(f'<circle cx="{x}" cy="{y}" r="20" fill="{NAVY}"/>')
    parts.append(f'<text x="{x}" y="{y + 5}" font-size="14" fill="white" text-anchor="middle">s{n}</text>')

# leaves: bar of height proportional to mu_l(x)
bar_max = 95
for k in range(8):
    x, y = pos[7 + k]
    p = reach[7 + k]
    h = bar_max * p / max(reach[7 + j] for j in range(8))
    parts.append(f'<rect x="{x - 22}" y="{y - 14}" width="44" height="28" rx="6" fill="{ICE}" stroke="{SKY}"/>')
    parts.append(f'<text x="{x}" y="{y + 5}" font-size="14" fill="{NAVY}" text-anchor="middle">ℓ{k + 1}</text>')
    base = y + 22 + bar_max
    t0 = 0.6 + 1.2 * 3
    parts.append(
        f'<rect x="{x - 18}" y="{base}" width="36" height="0" fill="{ORANGE}">'
        f'<animate attributeName="height" dur="{DUR}s" repeatCount="indefinite" '
        f'values="0;0;{h:.1f};{h:.1f};0" keyTimes="{kt(t0, t0 + 1.0)}"/>'
        f'<animate attributeName="y" dur="{DUR}s" repeatCount="indefinite" '
        f'values="{base};{base};{base - h:.1f};{base - h:.1f};{base}" keyTimes="{kt(t0, t0 + 1.0)}"/></rect>')
    parts.append(
        f'<text x="{x}" y="{base + 20}" font-size="14" fill="{NAVY}" text-anchor="middle" opacity="0">{p:.2f}'
        f'<animate attributeName="opacity" dur="{DUR}s" repeatCount="indefinite" '
        f'values="0;0;1;1;0" keyTimes="{kt(t0 + 0.6, t0 + 1.0)}"/></text>')

# input label
parts.append(f'<text x="450" y="22" font-size="17" fill="{NAVY}" text-anchor="middle" '
             f'font-weight="bold">input x</text>')
parts.append(f'<text x="880" y="22" font-size="14" fill="#5a6b7d" text-anchor="end">bars: leaf weights μℓ(x), sum = 1</text>')
parts.append("</svg>")

out = Path(__file__).parent / "figures" / "soft_routing.svg"
out.write_text("\n".join(parts))
print("wrote", out, "leaf sum", sum(reach[7 + k] for k in range(8)))
