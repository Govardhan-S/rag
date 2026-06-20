from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = Path("pdf_dataset")
OUTPUT_DIR.mkdir(exist_ok=True)

# Use a simple built-in font if available.
try:
    font = ImageFont.truetype("arial.ttf", size=16)
except OSError:
    font = ImageFont.load_default()

SAMPLE_TEXTS = [
    "This is sample document 1.\n\n" +
    "This dataset is generated for testing the RAG pipeline. "
    "The PDF contains multiple lines of text so that the parser can read it correctly.\n\n" * 4,

    "Document 2 contains another example of text.\n\n" +
    "Use this file to validate PDF extraction, chunking, embeddings, and retrieval.\n\n" * 5,

    "Third sample document.\n\n" +
    "This file includes a few short paragraphs.\n\n" * 6,

    "Analytics report sample.\n\n" +
    "The goal of this PDF dataset is to provide clean, readable content for the RAG demo.\n\n" * 5,

    "Final sample document.\n\n" +
    "Each PDF is generated with Pillow and saved in the pdf_dataset folder.\n\n" * 6,
]

PAGE_SIZE = (595, 842)  # A4 size in points at 72 DPI
MARGIN = 40
LINE_HEIGHT = 24


def render_text_to_pages(text: str, page_size, margin, line_height, font):
    lines = text.split("\n")
    pages = []
    current_lines = []
    max_lines_per_page = (page_size[1] - 2 * margin) // line_height

    for line in lines:
        current_lines.append(line)
        if len(current_lines) >= max_lines_per_page:
            pages.append(current_lines)
            current_lines = []

    if current_lines:
        pages.append(current_lines)

    images = []
    for page_lines in pages:
        image = Image.new("RGB", page_size, "white")
        draw = ImageDraw.Draw(image)
        y = margin
        for line in page_lines:
            draw.text((margin, y), line, fill="black", font=font)
            y += line_height
        images.append(image)

    return images


def create_pdf(filename: Path, contents: str):
    pages = render_text_to_pages(contents, PAGE_SIZE, MARGIN, LINE_HEIGHT, font)
    if not pages:
        return

    first_page, *remaining_pages = pages
    first_page.save(
        filename,
        format="PDF",
        save_all=True,
        append_images=remaining_pages,
    )


if __name__ == "__main__":
    print(f"Generating PDF dataset in {OUTPUT_DIR.absolute()}")
    for index, sample_text in enumerate(SAMPLE_TEXTS, start=1):
        output_path = OUTPUT_DIR / f"sample_document_{index}.pdf"
        create_pdf(output_path, sample_text)
        print(f"Created {output_path}")

    print("PDF dataset generation complete.")
