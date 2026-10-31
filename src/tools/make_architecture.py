"""Generate docs/architecture.drawio and docs/architecture.svg from one layout spec.

    python src/tools/make_architecture.py [OUT_DIR]   # default: <repo>/docs

Stdlib only. Both files come from the same node/edge table below (N, E), so the draw.io
source and the SVG embedded in docs/writeup.md never drift apart. Edit N/E, then rerun.
Open the .drawio file at https://app.diagrams.net to edit it by hand.
"""
import html
import sys
from pathlib import Path

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2] / "docs"
W, H = 980, 268
OFF, ON, DEC, CH, FB, LANE = "#dae8fc", "#d5e8d4", "#fff2cc", "#e1d5e7", "#f8cecc", "#f5f5f5"

# id: (x, y, w, h, label, fill, shape)
N = {
    "lane_off": (10, 6, 960, 92, "(a) OFFLINE ground truth (once; macOS teacher tools, never classifier features)", LANE, "lane"),
    "v0": (25, 32, 140, 60, "sub_category prior\nGT v0 (22 flagged)", OFF, "box"),
    "teach": (200, 32, 175, 60, "teacher signals\nWhisper/srt · cuts ·\noptical flow · OCR", OFF, "box"),
    "relabel": (410, 32, 160, 60, "anchor relabel\n5 features, margin 0.7", OFF, "box"),
    "review": (605, 32, 150, 60, "manual review\n6 videos (4/6 agree)", OFF, "box"),
    "v1": (790, 32, 160, 60, "GT v1 18/27/21/24\n→ train logreg (JSON)", OFF, "box"),
    "lane_on": (10, 108, 960, 154, "(b) ONLINE classification (stateless CPU workers, 1 core each)", LANE, "lane"),
    "queue": (25, 152, 100, 60, "ingestion\nqueue\n(video URI)", ON, "box"),
    "worker": (155, 137, 330, 90, "CPU worker\nffprobe → window schedule → ffmpeg window decode\n→ 17 cheap features → logistic regression\n≈ 4.1 CPU-s / video, ≈ 0.01 ms inference", ON, "box"),
    "conf": (515, 147, 120, 70, "max prob\n≥ 0.5 ?", DEC, "rhombus"),
    "router": (680, 127, 130, 40, "chunker router", CH, "box"),
    "c1": (840, 120, 120, 22, "semantic transcript", CH, "small"),
    "c2": (840, 146, 120, 22, "visual keyframe", CH, "small"),
    "c3": (840, 172, 120, 22, "action peak", CH, "small"),
    "c4": (840, 198, 120, 22, "shot / scene", CH, "small"),
    "fb": (680, 220, 280, 40, "fallback: shot / scene + transcript", FB, "box"),
}
# (src, dst, label, dashed)
E = [("v0", "teach", "", False), ("teach", "relabel", "", False), ("relabel", "review", "", False),
     ("review", "v1", "", False), ("v1", "worker", "model", True), ("queue", "worker", "", False),
     ("worker", "conf", "", False), ("conf", "router", "yes", False), ("conf", "fb", "no", False),
     ("router", "c1", "", False), ("router", "c2", "", False), ("router", "c3", "", False), ("router", "c4", "", False)]


def anchor(n, side):
    x, y, w, h = N[n][:4]
    return {"r": (x + w, y + h / 2), "l": (x, y + h / 2), "t": (x + w / 2, y), "b": (x + w / 2, y + h)}[side]


def route(s, d):
    """Return polyline points for an edge."""
    if (s, d) == ("v1", "worker"):
        a = anchor("v1", "b")
        return [a, (a[0], 103), (450, 103), (450, 137)]
    if s == "conf" and d == "router":
        a, b = anchor("conf", "t"), anchor("router", "l")
        return [a, (a[0], b[1]), b]
    if s == "conf" and d == "fb":
        a, b = anchor("conf", "b"), anchor("fb", "l")
        return [a, (a[0], b[1]), b]
    if s == "router":
        a, b = anchor("router", "r"), anchor(d, "l")
        return [a, (825, a[1]), (825, b[1]), b]
    return [anchor(s, "r"), anchor(d, "l")]


def svg():
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
         'font-family="Helvetica, Arial, sans-serif">',
         '<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
         'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#333"/></marker></defs>',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
    for k, (x, y, w, h, label, fill, shape) in N.items():
        if shape == "lane":
            o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="#999" stroke-dasharray="4 3"/>')
            o.append(f'<text x="{x + 10}" y="{y + 18}" font-size="12" font-weight="bold" fill="#444">{html.escape(label)}</text>')
            continue
        if shape == "rhombus":
            cx, cy = x + w / 2, y + h / 2
            o.append(f'<polygon points="{cx},{y} {x + w},{cy} {cx},{y + h} {x},{cy}" fill="{fill}" stroke="#b58900"/>')
        else:
            o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="5" fill="{fill}" stroke="#666"/>')
        lines = label.split("\n")
        fs = 10 if shape == "small" else 11
        y0 = y + h / 2 - (len(lines) - 1) * (fs + 2) / 2 + fs / 3
        for i, ln in enumerate(lines):
            o.append(f'<text x="{x + w / 2}" y="{y0 + i * (fs + 2):.1f}" font-size="{fs}" text-anchor="middle" fill="#111">{html.escape(ln)}</text>')
    for s, d, label, dashed in E:
        pts = route(s, d)
        dash = ' stroke-dasharray="5 4"' if dashed else ""
        o.append(f'<polyline points="{" ".join(f"{x},{y}" for x, y in pts)}" fill="none" stroke="#333" stroke-width="1.3"{dash} marker-end="url(#a)"/>')
        if label:
            (x1, y1), (x2, y2) = pts[0], pts[1]
            o.append(f'<text x="{(x1 + x2) / 2 + 5}" y="{(y1 + y2) / 2 - 3}" font-size="10" fill="#333">{label}</text>')
    o.append("</svg>")
    return "\n".join(o)


def drawio():
    cells = ['<mxCell id="0"/>', '<mxCell id="1" parent="0"/>']
    for k, (x, y, w, h, label, fill, shape) in N.items():
        style = {"lane": f"rounded=1;dashed=1;fillColor={fill};strokeColor=#999999;align=left;verticalAlign=top;spacingLeft=8;fontStyle=1;",
                 "rhombus": f"rhombus;whiteSpace=wrap;html=1;fillColor={fill};strokeColor=#b58900;",
                 "small": f"rounded=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor=#666666;fontSize=10;",
                 "box": f"rounded=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor=#666666;"}[shape]
        val = html.escape(label).replace("\n", "&lt;br&gt;")
        cells.append(f'<mxCell id="{k}" value="{val}" style="{style}" vertex="1" parent="1">'
                     f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
    for i, (s, d, label, dashed) in enumerate(E):
        pts = route(s, d)[1:-1]
        style = "edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;" + ("dashed=1;" if dashed else "")
        wp = "".join(f'<mxPoint x="{x}" y="{y}"/>' for x, y in pts)
        points = '<Array as="points">' + wp + "</Array>" if wp else ""
        cells.append(f'<mxCell id="e{i}" value="{label}" style="{style}" edge="1" parent="1" source="{s}" target="{d}">'
                     f'<mxGeometry relative="1" as="geometry">{points}</mxGeometry></mxCell>')
    return ('<mxfile host="app.diagrams.net"><diagram id="arch" name="Architecture">'
            f'<mxGraphModel dx="1000" dy="400" grid="1" gridSize="10" pageWidth="{W}" pageHeight="{H}"><root>'
            + "".join(cells) + "</root></mxGraphModel></diagram></mxfile>\n")


(OUT / "architecture.svg").write_text(svg())
(OUT / "architecture.drawio").write_text(drawio())
print("wrote", OUT / "architecture.svg", OUT / "architecture.drawio")
