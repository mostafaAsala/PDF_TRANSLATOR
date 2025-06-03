import fitz  # PyMuPDF
import os

# Output path and text
output_path = "output_text_with_local_font.pdf"
text = "Test with special symbols: 20∆4 ± ❤️ 你好"

# Path to a valid .ttf font (adjust for your OS)
font_path = {
    "win": "C:/Windows/Fonts/arial.ttf",
    "linux": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "mac": "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
}

if os.name == "nt":
    font_file = font_path["win"]
elif os.path.exists(font_path["linux"]):
    font_file = font_path["linux"]
else:
    font_file = font_path["mac"]

# Create PDF and page
doc = fitz.open()
page = doc.new_page()

# Insert the text using local font
page.insert_text(
    (72, 100),                  # position in points
    text,
    fontsize=16,
    fontname="custom",         # any name you want
    fontfile=font_file,        # full path to .ttf file
    color=(0, 0, 0)            # black
)

# Save and close
doc.save(output_path)
doc.close()

print(f"Saved to {output_path}")
