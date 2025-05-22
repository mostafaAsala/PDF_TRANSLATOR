import io

import fitz  # PyMuPDF
from PIL import Image
#from deep_translator import GoogleTranslator
# Load model directly
print("loading packages")
from transformers import AutoTokenizer, MarianMTModel, AutoModelForSeq2SeqLM

src = "zh"  # source language
trg = "en"  # target language
print("loading model")
model_name = f"Helsinki-NLP/opus-mt-{src}-{trg}"



tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

#model = MarianMTModel.from_pretrained(model_name)
#tokenizer = AutoTokenizer.from_pretrained(model_name)
print("begin translating, ")


def int_to_rgb(color_int):
    """Convert integer color to normalized RGB tuple."""
    r = ((color_int >> 16) & 255) / 255
    g = ((color_int >> 8) & 255) / 255
    b = (color_int & 255) / 255
    return (r, g, b)


def extract_image_with_transparency(doc, xref):
    # Step 1: Get image object dictionary
    obj = doc.xref_object(xref, compressed=False)

    # Step 2: Check if SMask is used (soft transparency mask)
    if "/SMask" not in obj:
        # No soft mask — fallback to standard extraction
        img_info = doc.extract_image(xref)
        img = Image.open(io.BytesIO(img_info["image"])).convert("RGBA")
        return img

    # Step 3: Get image and soft mask xrefs
    img_info = doc.extract_image(xref)
    img_data = img_info["image"]
    img = Image.open(io.BytesIO(img_data)).convert("RGB")

    # Parse SMask reference
    smask_line = [line for line in obj.splitlines() if "/SMask" in line][0]

    smask_xref = int(smask_line.split()[1])
    smask_info = doc.extract_image(smask_xref)
    smask_data = smask_info["image"]
    smask = Image.open(io.BytesIO(smask_data)).convert("L")  # grayscale alpha mask

    # Step 4: Combine RGB image with alpha channel
    img.putalpha(smask)

    return img  # RGBA image

def sanitize_text(text):
    if not isinstance(text, str):
        try:
            text = text.decode('utf-8', errors='ignore')
        except Exception:
            text = str(text)
    # Remove non-printable characters and control characters
    return ''.join(c for c in text if c.isprintable())

def translate_pdf_text_precise(input_pdf_path, output_pdf_path, source_lang='auto', target_lang='en'):
    # Open the original document
    doc = fitz.open(input_pdf_path)

    # Create a new empty document for the translation
    translated_doc = fitz.open()

    # First, collect all text that needs translation from all pages
    all_text_to_translate = []
    text_metadata = []  # Store metadata for each text element

    print("Collecting all text for translation...")
    for page_number, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if block['type'] == 0:  # Text block
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if text:
                            # Store the text and its metadata
                            all_text_to_translate.append(text)
                            text_metadata.append({
                                'page_number': page_number,
                                'bbox': span["bbox"],
                                'font_size': span["size"],
                                'color_int': span["color"],
                                'text_font': span['font'],
                                'flags': span['flags']
                            })

    # Translate all text at once
    print(f"Translating {len(all_text_to_translate)} text elements...")
    all_translations = []

    # Process in batches if needed to avoid memory issues
    batch_size = 50  # Adjust based on your model's capacity
    for i in range(0, len(all_text_to_translate), batch_size):
        batch_texts = all_text_to_translate[i:i+batch_size]
        try:
            # Prepare batch for translation
            tokenizer.src_lang = "zh_CN"
            batch_translations = []

            # Handle empty strings first
            non_empty_texts = []
            non_empty_indices = []

            for idx, text in enumerate(batch_texts):
                if text == '':
                    batch_translations.append('■  ')
                else:
                    non_empty_texts.append(text)
                    non_empty_indices.append(idx)

            if non_empty_texts:
                try:
                    print(f"Batch translating {len(non_empty_texts)} texts at once")
                    # Encode all texts in a single batch
                    encoded_batch = tokenizer(non_empty_texts, padding=True, return_tensors="pt")

                    # Generate translations for the entire batch at once
                    generated_tokens = model.generate(
                        **encoded_batch,
                        forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"]
                    )

                    # Decode all translations at once
                    translated_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                    print(f" done Batch translating {len(non_empty_texts)} texts at once")
                    # Insert translations back in the correct positions
                    for batch_idx, original_idx in enumerate(non_empty_indices):
                        if batch_idx < len(translated_texts):
                            # Insert at the position corresponding to the original text
                            while len(batch_translations) <= original_idx:
                                batch_translations.append(None)  # Pad if needed
                            batch_translations[original_idx] = translated_texts[batch_idx]
                        else:
                            # Fallback if something went wrong with the batch indices
                            batch_translations.append(non_empty_texts[batch_idx])

                    # Fill any remaining None values with original text as fallback
                    for idx, translation in enumerate(batch_translations):
                        if translation is None and idx < len(batch_texts):
                            batch_translations[idx] = batch_texts[idx]

                except Exception as e:
                    print(f"Batch translation error: {e}")
                    # If batch translation fails, use original texts as fallback
                    for idx, original_idx in enumerate(non_empty_indices):
                        while len(batch_translations) <= original_idx:
                            batch_translations.append(None)
                        batch_translations[original_idx] = non_empty_texts[idx]

            all_translations.extend(batch_translations)
            print(f"Translated batch {i//batch_size + 1}/{(len(all_text_to_translate) + batch_size - 1)//batch_size}")

        except Exception as e:
            print(f"Batch translation error: {e}")
            # If batch fails, add original text as fallback
            all_translations.extend(batch_texts[len(all_translations) - i:])

    # Now create pages and add content
    page_num = 1

    for page_number, page in enumerate(doc):
        # Create a new blank page with the same dimensions
        new_page = translated_doc.new_page(width=page.rect.width, height=page.rect.height)
        print(f"Processing page {page_number+1}/{len(doc)}, dimensions: {page.rect.width} x {page.rect.height}")

        # Get text blocks for this page (needed for structure)
        blocks = page.get_text("dict")["blocks"]

        # 1. First, extract and redraw all drawings (lines, shapes, etc.)
        drawings = page.get_drawings()
        for drawing in drawings:
            shape = new_page.new_shape()
            length = len(drawing["items"])
            for i in range(length):
                item = drawing["items"][i]
                if item[0] == "l":  # line
                    p1, p2 = item[1], item[2]
                    # Only draw the line if it's not the extra connection between first and last points
                    if i != length:  # Skip the line if it's the closing one
                        shape.draw_line(p1, p2)
                elif item[0] == "re":  # rectangle
                    r = item[1]
                    shape.draw_rect(r)
                elif item[0] == "qu":  # curve
                    p1, p2, p3 = item[1], item[2], item[3]
                    shape.draw_line(p1, p3)  # approximate curve as line
                elif item[0] == "c":  # curve
                    shape.draw_bezier(item[1], item[2], item[3], item[4])
                else:
                    print(f"Unknown drawing item type: {item[0]}")

            shape.finish(width=drawing['width'],
                         color=drawing['color'],
                         closePath=drawing['closePath'],
                         fill=drawing['fill'],
                         dashes=drawing['dashes'],
                         stroke_opacity=1 if drawing.get("stroke_opacity", 1) is None else drawing.get("stroke_opacity", 1))
            shape.commit()

        # 2. Extract and copy all images
        image_list = page.get_image_info(hashes=False, xrefs=True)
        for idx, img_info in enumerate(image_list):
            xref = img_info['xref']
            image = extract_image_with_transparency(doc, xref)

            # Extract the image bytes and other info
            base_image = doc.extract_image(xref)
            bbox = img_info['bbox']

            image = image.resize((int(image.width * 0.8), int(image.height * 0.8)), resample=Image.LANCZOS)

            # Save modified image to bytes
            output = io.BytesIO()
            image.save(output, format="PNG", dpi=(72, 72), optimize=True)
            output.seek(0)
            modified_img_bytes = output.read()

            image.save(f'images/extracted_image{idx}.png', 'PNG')  # Save as PNG
            # Insert the image using raw bytes (preserves alpha if present)
            new_page.insert_image(bbox, stream=modified_img_bytes, keep_proportion=False)

        # 3. Add translated text for this page
        for block in blocks:
            if block['type'] == 0:  # Text block
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if text:
                            # Find the corresponding translation
                            meta_indices = [i for i, meta in enumerate(text_metadata)
                                           if meta['page_number'] == page_number and meta['bbox'] == span["bbox"]]

                            if meta_indices:
                                meta_index = meta_indices[0]
                                translated_text = all_translations[meta_index]

                                # Get metadata
                                meta = text_metadata[meta_index]
                                bbox = meta['bbox']
                                font_size = meta['font_size']
                                color_int = meta['color_int']
                                text_font = 'helv'  # Using helv as standard font

                                # Process and insert the translated text
                                translated_text_clean = sanitize_text(translated_text)
                                text_width = fitz.get_text_length(translated_text_clean, fontname='helv',
                                                                 fontsize=font_size - 2)
                                bbox_width = bbox[2] - bbox[0]

                                scale_x = bbox_width / text_width if text_width > 0 else 1.0

                                color_rgb = int_to_rgb(color_int)
                                pivot = fitz.Point(bbox[0], bbox[1])
                                mat = fitz.Matrix(scale_x, 1)
                                morph = (pivot, mat)

                                # Insert translated text at the same position
                                new_page.insert_text(
                                    (bbox[0], bbox[1] + 10),
                                    translated_text,
                                    fontsize=font_size - 2,
                                    fontname=text_font,  # standard font
                                    color=color_rgb,
                                    morph=morph,
                                )
            elif block['type'] == 1:  # image
                # Images are already handled above
                pass
            else:
                print(f"Unknown block type: {block}")

        page_num += 1

    translated_doc.save(output_pdf_path)
    translated_doc.save("compressed_output.pdf", garbage=4, deflate=True, clean=True)
    print(f"Translated PDF saved to {output_pdf_path}")


if __name__ == "__main__":
    input_pdf = r"C:\Users\161070\Downloads\o4ybafz0i7eafqw6abn61smnd1i676.pdf"
    #input_pdf = r"C:\Users\161070\Downloads\2409191731_hikvision-ds-ipc-b12hv3-ia-poe4mm_c41359959.pdf"
    output_pdf = "translated_text_preserved.pdf"
    translate_pdf_text_precise(input_pdf, output_pdf)
