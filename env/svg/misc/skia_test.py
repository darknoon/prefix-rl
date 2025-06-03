from skia import MemoryStream, SVGDOM, ImageInfo, Surface, kPNG

# 1) Load SVG bytes from disk
svg_bytes = open("env/svg/svgo_regression_300.svg", "rb").read()
stream = MemoryStream(svg_bytes)

# 2) Parse into SkSVGDOM
dom = SVGDOM.MakeFromStream(stream)
if dom is None:
    raise RuntimeError("Failed to parse SVG")


size = dom.containerSize()
width, height = int(size.width()), int(size.height())
info = ImageInfo.MakeN32Premul(width, height)
surface = Surface.MakeRaster(info)
canvas = surface.getCanvas()

dom.render(canvas)

img = surface.makeImageSnapshot()
img.save("output.png", kPNG)
print("Wrote output.png")
