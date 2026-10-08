# ABOUTME: Regenerates the two tiny Chinese OCR test images (simplified and traditional) used by tests/test_reader.py.
# ABOUTME: Needs Pillow and the macOS fonts named below; the generated PNGs are committed, so tests never run this.
import os

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FONTS = "/System/Library/Fonts"


def render(text: str, font_file: str, out_name: str) -> None:
    font = ImageFont.truetype(os.path.join(FONTS, font_file), 40, index=0)
    image = Image.new("L", (460, 90), 255)
    ImageDraw.Draw(image).text((20, 20), text, font=font, fill=0)
    image.save(os.path.join(HERE, out_name), optimize=True)


if __name__ == "__main__":
    render("发票 总计 1234", "Hiragino Sans GB.ttc", "zh-sim.png")
    render("發票 總計 1234", "STHeiti Medium.ttc", "zh-tra.png")
