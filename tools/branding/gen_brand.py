#!/usr/bin/env python3
"""Generate the Vokra brand assets into ``assets/branding/``.

Offline, developer-side tool. Nothing here enters the runtime or the Cargo
graph (NFR-DS-02): the repository ships the *generated* files, and neither
the build nor CI needs this script, ``rsvg-convert``, or ``fontTools``.

THE MARK
--------
A Roman inscriptional ``V``. It is not a drawn letter sitting on a tile — it
is the void left after cutting a V-section groove into a solid block, which
is how Roman inscriptions were actually made.

The stroke contrast (left downstroke 116, right upstroke 72) is not styling.
Cutting stone with a chisel widens the groove differently depending on the
direction of the cut, so the downstroke comes out heavier. The terminals are
cut horizontally and the strokes flare into them. Every line in the outline
is a consequence of the tool; none was added to look nice.

This maps onto what the project claims about itself:

    depth over breadth   the form follows the tool and carries nothing
                         that is not structural
    zero dependencies    one block: no parts, no seams, nothing attached
    no graph ingestion   the opposite of a node-and-edge mesh; an
                         unbreakable single face
    no silent fallback   every edge is a straight line, no gradients, no
                         blur, nothing that fades out ambiguously
    runs everywhere      one monochrome silhouette that is the same object
                         at 24 px and at 1280 px, with no simplified variant

COLOUR
------
Ink ``#15131A`` / paper ``#F6F4F0`` / minium ``#C8452B``.

Minium (red lead) is the pigment Roman carvers packed into the cut grooves so
the inscription could be read from a distance. The accent is therefore a
continuation of how the mark is made rather than decoration applied to it.

TYPE
----
Wordmark set in Optima. Zapf drew it in 1950 from Roman gravestone lettering
in the Basilica di Santa Croce, Florence — strokes that widen towards their
terminals without ending in a serif. That is the same construction as the
flare on this mark, so the pairing is a shared lineage rather than a
coincidence of taste.

**Optima is used for rasterised output only.** Committing font outlines as
SVG paths would redistribute the typeface, which its licence does not allow;
setting text into a PNG is ordinary use. Every SVG this script emits is the
mark alone, with no glyph data. A custom-drawn logotype, if one is ever
wanted, is separate work.

GEOMETRY
--------
All dimensions are constants at the top of this file. The shrink factor that
keeps the mark inside the circular-crop safe area is *derived* from that
safe area, never hand-tuned: adjusting a weight or a serif by hand and then
forgetting to re-check the corners is exactly how a clipped avatar ships.

REQUIREMENTS
------------
``rsvg-convert`` (librsvg) for rasterising, ``fonttools`` for measuring the
wordmark, and Optima installed at ``FONT_FILE``. macOS ships Optima; on other
platforms the SVG assets still generate but the two PNG compositions do not.

USAGE
-----
    python3 tools/branding/gen_brand.py
"""

import math
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "assets" / "branding"

CX = CY = 256.0
SAFE_R = 200.0  # circular-crop safe radius within the 512 canvas

INK = "#15131A"
PAPER = "#F6F4F0"
MINIUM = "#C8452B"

TOP, APEX = 118.0, 410.0
LO, THICK = 104.0, 116.0  # left = downstroke, heavy
RO, THIN = 408.0, 72.0  # right = upstroke, light
SERIF_EXT, SERIF_H = 14.0, 30.0  # terminal flare: width each side, and its height

FONT = "Optima"
FONT_FILE = "/System/Library/Fonts/Optima.ttc"


def outline(serif=True, fit=None):
    """Return the mark as one closed path plus its vertices.

    The flare is woven into the outline rather than bolted on as a rectangle.
    Laying a horizontal bar across a diagonal stroke leaves a visible step at
    the junction, which reads as a glitch instead of a serif.

    ``fit`` defaults to the largest scale that keeps every vertex inside
    ``SAFE_R``.
    """
    ext, sh = (SERIF_EXT, SERIF_H) if serif else (0.0, 0.001)
    li, ri = LO + THICK, RO - THIN
    dy = APEX - TOP
    kl = (CX - LO) * sh / dy  # how far the left edge travels over the flare
    kr = (CX - RO) * sh / dy  # the right edge travels the other way
    t = (ri - li) / ((CX - LO) - (CX - RO))
    # The inner vertex lands right of the centre line. That asymmetry is what
    # makes the shape read as a letter rather than as a geometric chevron.
    inner = (li + (CX - LO) * t, TOP + dy * t)

    pts = [
        (LO - ext, TOP), (LO + kl, TOP + sh),
        (CX, APEX),
        (RO + kr, TOP + sh), (RO + ext, TOP),
        (ri + ext, TOP), (ri + kr, TOP + sh),
        inner,
        (li + kl, TOP + sh), (li - ext, TOP),
    ]
    if fit is None:
        fit = SAFE_R / max(math.hypot(x - CX, y - CY) for x, y in pts)
    pts = [(CX + (x - CX) * fit, CY + (y - CY) * fit) for x, y in pts]
    d = "M %.1f %.1f " % pts[0] + " ".join("L %.1f %.1f" % p for p in pts[1:]) + " Z"
    return d, pts


D_SERIF, PTS_SERIF = outline(True)
D_PLAIN, _ = outline(False)

# The 512 canvas carries padding for the circular crop, so typesetting against
# the canvas leaves a hole between the mark and the words. Place against the
# real bounding box instead.
BB_X0 = min(x for x, _ in PTS_SERIF)
BB_X1 = max(x for x, _ in PTS_SERIF)
BB_Y0 = min(y for _, y in PTS_SERIF)
BB_Y1 = max(y for _, y in PTS_SERIF)


def mark_width(height):
    return (BB_X1 - BB_X0) * height / (BB_Y1 - BB_Y0)


def place_mark(x, y, height, fill):
    """Put the mark's bounding box at ``(x, y)`` with the given height."""
    s = height / (BB_Y1 - BB_Y0)
    return (
        f'<g transform="translate({x - BB_X0 * s:.2f},{y - BB_Y0 * s:.2f}) '
        f'scale({s:.5f})"><path d="{D_SERIF}" fill="{fill}"/></g>'
    )


def text_width(s, size, tracking=0.0, style="Regular"):
    """Measure a string set in Optima, in user units.

    Guessing the lockup canvas width by eye is not reproducible, so derive it
    from the font's advance widths. librsvg applies ``letter-spacing`` after
    every glyph including the last, so the trailing gap is counted too.
    """
    from fontTools.ttLib import TTCollection

    for f in TTCollection(FONT_FILE).fonts:
        if f["name"].getDebugName(2) == style:
            upem = f["head"].unitsPerEm
            cmap = f.getBestCmap()
            hmtx = f["hmtx"]
            total = sum(hmtx[cmap[ord(c)]][0] for c in s if ord(c) in cmap)
            return total * size / upem + tracking * len(s)
    raise RuntimeError(f"Optima {style} not found in {FONT_FILE}")


def tile(bg, fg, d=D_SERIF):
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" '
        'width="512" height="512" role="img">'
        f'<rect width="512" height="512" fill="{bg}"/><path d="{d}" fill="{fg}"/></svg>'
    )


def bare(d=D_SERIF, color="currentColor"):
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" '
        f'width="512" height="512" role="img" color="{INK}">'
        f'<path d="{d}" fill="{color}"/></svg>'
    )


TAGLINE = "Speech-first inference runtime in Rust"
CAPABILITIES = "TTS · ASR · SPEECH-TO-SPEECH · VAD · APACHE-2.0"


def social(w=1280, h=640):
    """GitHub social preview.

    Everything sits in the middle: Twitter, Slack and others re-crop the
    image to their own aspect ratios, and edge-anchored content loses its
    head in the process.
    """
    mh = 172
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
<rect width="{w}" height="{h}" fill="{INK}"/>
{place_mark((w - mark_width(mh)) / 2, 104, mh, MINIUM)}
<text x="{w / 2}" y="406" text-anchor="middle" font-family="{FONT}" font-size="100"
      letter-spacing="22" fill="{PAPER}">VOKRA</text>
<rect x="{w / 2 - 132}" y="446" width="264" height="2" fill="{MINIUM}"/>
<text x="{w / 2}" y="506" text-anchor="middle" font-family="{FONT}" font-size="31"
      letter-spacing="1.5" fill="{PAPER}" opacity="0.86">{TAGLINE}</text>
<text x="{w / 2}" y="556" text-anchor="middle" font-family="{FONT}" font-size="21"
      letter-spacing="3.4" fill="{PAPER}" opacity="0.5">{CAPABILITIES}</text>
</svg>"""


def lockup(h=272, pad=88):
    """Horizontal lockup for README headers. Width comes from measured text."""
    mh = 116
    mx, my = pad, (h - mh) / 2
    tx = mx + mark_width(mh) + 46
    w = round(tx + max(text_width("VOKRA", 76, 15), text_width(TAGLINE, 23, 1.2)) + pad)
    body = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
<rect width="{w}" height="{h}" fill="{INK}"/>
{place_mark(mx, my, mh, MINIUM)}
<text x="{tx:.1f}" y="142" font-family="{FONT}" font-size="76" letter-spacing="15"
      fill="{PAPER}">VOKRA</text>
<text x="{tx + 3:.1f}" y="186" font-family="{FONT}" font-size="23" letter-spacing="1.2"
      fill="{PAPER}" opacity="0.64">{TAGLINE}</text>
</svg>"""
    return body, w


SVGS = [
    ("vokra-mark.svg", bare()),
    ("vokra-mark-plain.svg", bare(D_PLAIN)),
    ("vokra-mark-minium.svg", bare(color=MINIUM)),
    ("vokra-avatar.svg", tile(INK, MINIUM)),
    ("vokra-avatar-mono.svg", tile(INK, PAPER)),
    ("vokra-avatar-light.svg", tile(PAPER, INK)),
]

PNGS = [
    ("vokra-avatar.svg", "vokra-avatar-1024.png", 1024),
    ("vokra-avatar.svg", "vokra-avatar-512.png", 512),
    ("vokra-avatar.svg", "vokra-icon-64.png", 64),
    ("vokra-avatar.svg", "vokra-icon-32.png", 32),
    ("vokra-avatar-mono.svg", "vokra-avatar-mono-512.png", 512),
    ("vokra-avatar-light.svg", "vokra-avatar-light-512.png", 512),
]


def render(src, dst, width):
    subprocess.run(["rsvg-convert", "-w", str(width), str(src), "-o", str(dst)], check=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    r = max(math.hypot(x - CX, y - CY) for x, y in PTS_SERIF)
    assert r <= SAFE_R + 0.01, f"outermost vertex {r:.1f} escapes the {SAFE_R} safe area"
    print(f"circular-crop safe area: outermost r={r:.1f} / {SAFE_R:.0f} OK")

    for name, body in SVGS:
        (OUT / name).write_text(body, encoding="utf-8")

    lockup_svg, lockup_w = lockup()
    for stem, body, width in (
        ("vokra-social", social(), 1280),
        ("vokra-lockup", lockup_svg, lockup_w),
    ):
        tmp = OUT / f".{stem}.svg"
        tmp.write_text(body, encoding="utf-8")
        render(tmp, OUT / f"{stem}.png", width)
        tmp.unlink()

    for src, dst, width in PNGS:
        render(OUT / src, OUT / dst, width)

    print(f"wrote {len(SVGS)} SVG + {len(PNGS) + 2} PNG to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
