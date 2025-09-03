"""
Helper functions for generating SVG shapes for testing and visualization.
"""


def make_circle(
    cx=256,
    cy=256,
    r=100,
    stroke_width=3,
    fill="red",
    stroke="black",
    background="white",
    width=512,
    height=512,
):
    return f"""<svg width='{width}' height='{height}' xmlns='http://www.w3.org/2000/svg'>
    <rect width='{width}' height='{height}' fill='{background}' />
    <circle cx='{cx}' cy='{cy}' r='{r}' stroke='{stroke}' stroke-width='{stroke_width}' fill='{fill}' />
</svg>"""


def make_rectangle(
    x=100,
    y=100,
    width=200,
    height=150,
    stroke_width=3,
    fill="blue",
    stroke="black",
    background="white",
    canvas_width=512,
    canvas_height=512,
):
    return f"""<svg width='{canvas_width}' height='{canvas_height}' xmlns='http://www.w3.org/2000/svg'>
    <rect width='{canvas_width}' height='{canvas_height}' fill='{background}' />
    <rect x='{x}' y='{y}' width='{width}' height='{height}' stroke='{stroke}' stroke-width='{stroke_width}' fill='{fill}' />
</svg>"""


def make_line(
    x1=100,
    y1=100,
    x2=200,
    y2=200,
    stroke_width=3,
    stroke="black",
    background="white",
    canvas_width=512,
    canvas_height=512,
):
    return f"""<svg width='{canvas_width}' height='{canvas_height}' xmlns='http://www.w3.org/2000/svg'>
    <rect width='{canvas_width}' height='{canvas_height}' fill='{background}' />
    <line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='{stroke}' stroke-width='{stroke_width}' />
</svg>"""
