import io
import argparse
import os
import time
from turtle import pd

import fitz  # PyMuPDF
from PIL import Image
from googletrans import Translator as GoogleTranslator
from deep_translator import GoogleTranslator as DeepGoogleTranslator
# Load model directly
print("loading packages")
from transformers import AutoTokenizer, MarianMTModel, AutoModelForSeq2SeqLM
import pandas as pd
src = "zh"  # source language
trg = "en"  # target language
print("loading model")
model_name = f"Helsinki-NLP/opus-mt-{src}-{trg}"

# Initialize Google Translator
google_translator = GoogleTranslator()

# Initialize Deep Translator
deep_translator = None

# Only load the model if needed (will be loaded on demand)
tokenizer = None
model = None

def load_local_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        print("Loading local translation model...")
        tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        print("Local model loaded")

def load_deep_translator():
    # This function is kept for consistency with the other translator functions
    # but we don't need to initialize anything globally since we create a new
    # translator instance for each batch or individual translation
    print("Deep Translator will be initialized for each translation request")

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

def batch_translate_with_local_model(batch_texts):
    """Translate a batch of texts using the local model."""
    load_local_model()  # Ensure model is loaded

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
            print(f"Batch translating {len(non_empty_texts)} texts at once with local model")
            # Encode all texts in a single batch
            tokenizer.src_lang = "zh_CN"
            encoded_batch = tokenizer(non_empty_texts, padding=True, return_tensors="pt")

            # Generate translations for the entire batch at once
            generated_tokens = model.generate(
                **encoded_batch,
                forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"]
            )

            # Decode all translations at once
            translated_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            print(f"Done batch translating {len(non_empty_texts)} texts")

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

    return batch_translations

def batch_translate_with_google(batch_texts, source_lang='zh-cn', target_lang='en'):
    """Translate a batch of texts using Google Translate API."""
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
            print(f"Batch translating {len(non_empty_texts)} texts at once with Google Translate")

            # Google Translate can handle multiple texts at once in a single API call
            translations = google_translator.translate(non_empty_texts, src=source_lang, dest=target_lang)

            print(f"Done batch translating {len(non_empty_texts)} texts with Google Translate")

            # Insert translations back in the correct positions
            for batch_idx, original_idx in enumerate(non_empty_indices):
                if batch_idx < len(translations):
                    # Insert at the position corresponding to the original text
                    while len(batch_translations) <= original_idx:
                        batch_translations.append(None)  # Pad if needed
                    batch_translations[original_idx] = translations[batch_idx].text
                else:
                    # Fallback if something went wrong with the batch indices
                    batch_translations.append(non_empty_texts[batch_idx])

            # Fill any remaining None values with original text as fallback
            for idx, translation in enumerate(batch_translations):
                if translation is None and idx < len(batch_texts):
                    batch_translations[idx] = batch_texts[idx]

        except Exception as e:
            print(f"Google Translate batch error: {e}")
            print("Falling back to individual translation...")

            # If batch translation fails, try translating one by one
            translations = []
            for text in non_empty_texts:
                try:
                    result = google_translator.translate(text, src=source_lang, dest=target_lang)
                    translations.append(result.text)
                except Exception as e:
                    print(f"Error translating text '{text}': {e}")
                    translations.append(text)  # Use original as fallback
                # Add a small delay to avoid rate limiting

            # Insert translations back in the correct positions
            for batch_idx, original_idx in enumerate(non_empty_indices):
                if batch_idx < len(translations):
                    while len(batch_translations) <= original_idx:
                        batch_translations.append(None)
                    batch_translations[original_idx] = translations[batch_idx]
                else:
                    batch_translations.append(non_empty_texts[batch_idx])

            # Fill any remaining None values
            for idx, translation in enumerate(batch_translations):
                if translation is None and idx < len(batch_texts):
                    batch_translations[idx] = batch_texts[idx]

    return batch_translations

def batch_translate_with_deep(batch_texts, source_lang='zh-cn', target_lang='en'):
    """Translate a batch of texts using Deep Translator."""
    load_deep_translator()  # Ensure translator is loaded
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
            print(f"Batch translating {len(non_empty_texts)} texts at once with Deep Translator")

            # Deep Translator can handle multiple texts at once
            # Set up the translator with the correct source and target languages
            translator = DeepGoogleTranslator(source=source_lang, target=target_lang)
            translations = translator.translate_batch(non_empty_texts)

            print(f"Done batch translating {len(non_empty_texts)} texts with Deep Translator")

            # Insert translations back in the correct positions
            for batch_idx, original_idx in enumerate(non_empty_indices):
                if batch_idx < len(translations):
                    # Insert at the position corresponding to the original text
                    while len(batch_translations) <= original_idx:
                        batch_translations.append(None)  # Pad if needed
                    batch_translations[original_idx] = translations[batch_idx]
                else:
                    # Fallback if something went wrong with the batch indices
                    batch_translations.append(non_empty_texts[batch_idx])

            # Fill any remaining None values with original text as fallback
            for idx, translation in enumerate(batch_translations):
                if translation is None and idx < len(batch_texts):
                    batch_translations[idx] = batch_texts[idx]

        except Exception as e:
            print(f"Deep Translator batch error: {e}")
            print("Falling back to individual translation...")

            # If batch translation fails, try translating one by one
            translations = []
            for text in non_empty_texts:
                try:
                    # Set up the translator with the correct source and target languages
                    translator = DeepGoogleTranslator(source=source_lang, target=target_lang)
                    result = translator.translate(text)
                    translations.append(result)
                except Exception as e:
                    print(f"Error translating text '{text}': {e}")
                    translations.append(text)  # Use original as fallback
                # Add a small delay to avoid rate limiting
                time.sleep(0.5)

            # Insert translations back in the correct positions
            for batch_idx, original_idx in enumerate(non_empty_indices):
                if batch_idx < len(translations):
                    while len(batch_translations) <= original_idx:
                        batch_translations.append(None)
                    batch_translations[original_idx] = translations[batch_idx]
                else:
                    batch_translations.append(non_empty_texts[batch_idx])

            # Fill any remaining None values
            for idx, translation in enumerate(batch_translations):
                if translation is None and idx < len(batch_texts):
                    batch_translations[idx] = batch_texts[idx]

    return batch_translations

def translate_pdf_text_precise(input_pdf_path, output_pdf_path, source_lang='auto', target_lang='en', translator_type='local'):
    """
    Translate PDF text with precise positioning.

    Args:
        input_pdf_path: Path to the input PDF file
        output_pdf_path: Path to save the translated PDF
        source_lang: Source language code (default: 'auto')
        target_lang: Target language code (default: 'en')
        translator_type: Type of translator to use ('local', 'google', or 'deep')
    """
    # Open the original document
    doc = fitz.open(input_pdf_path)
    total_pages = len(doc)

    # Create a new empty document for the translation
    translated_doc = fitz.open()

    # Process pages in batches of 10
    save_interval = 10

    # Get base name and extension for intermediate saves
    import os
    base_name, ext = os.path.splitext(output_pdf_path)

    for page_batch_start in range(0, total_pages, save_interval):
        page_batch_end = min(page_batch_start + save_interval, total_pages)
        print(f"\n--- Processing pages {page_batch_start+1} to {page_batch_end} of {total_pages} ---\n")

        # Process each page in the current batch
        for page_number in range(page_batch_start, page_batch_end):
            page = doc[page_number]
            print(f"Processing page {page_number+1}/{total_pages}")

            # Collect text from this page
            page_text_to_translate = []
            page_text_metadata = []  # Store metadata for each text element

            print(f"Collecting text from page {page_number+1}...")
            blocks = page.get_text("dict")["blocks"]

            for block in blocks:
                if block['type'] == 0:  # Text block
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"].strip()
                            if text:
                                # Store the text and its metadata
                                page_text_to_translate.append(text)
                                page_text_metadata.append({
                                    'bbox': span["bbox"],
                                    'font_size': span["size"],
                                    'color_int': span["color"],
                                    'text_font': span['font'],
                                    'flags': span['flags']
                                })

            # Translate text for this page
            if page_text_to_translate:
                print(f"Translating {len(page_text_to_translate)} text elements from page {page_number+1} using {translator_type} translator...")
                page_translations = []

                # Process in batches if needed to avoid memory issues
                batch_size = 250  # Adjust based on your model's capacity
                for i in range(0, len(page_text_to_translate), batch_size):
                    batch_texts = page_text_to_translate[i:i+batch_size]
                    try:
                        # Choose translation method based on user selection
                        if translator_type.lower() == 'google':
                            batch_translations = batch_translate_with_google(batch_texts, source_lang, target_lang)
                        elif translator_type.lower() == 'deep':
                            batch_translations = batch_translate_with_deep(batch_texts, source_lang, target_lang)
                        else:  # Default to local model
                            batch_translations = batch_translate_with_local_model(batch_texts)

                        page_translations.extend(batch_translations)
                        print(f"Translated batch {i//batch_size + 1}/{(len(page_text_to_translate) + batch_size - 1)//batch_size}")

                    except Exception as e:
                        print(f"Batch translation error: {e}")
                        # If batch fails, add original text as fallback
                        page_translations.extend(batch_texts[len(page_translations) - i:])
            else:
                page_translations = []
                print(f"No text to translate on page {page_number+1}")

            # Create a new blank page with the same dimensions
            new_page = translated_doc.new_page(width=page.rect.width, height=page.rect.height)
            print(f"Building page {page_number+1}, dimensions: {page.rect.width} x {page.rect.height}")

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

                # Get the bounding box for the image
                bbox = img_info['bbox']

                image = image.resize((int(image.width * 0.8), int(image.height * 0.8)), resample=Image.LANCZOS)

                # Save modified image to bytes
                output = io.BytesIO()
                image.save(output, format="PNG", dpi=(72, 72), optimize=True)
                output.seek(0)
                modified_img_bytes = output.read()

                # Save image to file (optional)
                image_dir = 'images'
                if not os.path.exists(image_dir):
                    os.makedirs(image_dir)
                image.save(f'{image_dir}/page{page_number+1}_image{idx}.png', 'PNG')

                # Insert the image using raw bytes (preserves alpha if present)
                new_page.insert_image(bbox, stream=modified_img_bytes, keep_proportion=False)

            # 3. Add translated text for this page
            if page_text_to_translate:
                for idx, (text, meta) in enumerate(zip(page_text_to_translate, page_text_metadata)):
                    if idx < len(page_translations):
                        translated_text = page_translations[idx]

                        # Get metadata
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

        # Save the document after each batch of pages
        if page_batch_end >= page_batch_start + 1:  # Only save if at least one page was processed
            batch_output_path = f"{base_name}_pages_{page_batch_start+1}_to_{page_batch_end}{ext}"
            translated_doc.save(batch_output_path,garbage=4, deflate=True, clean=True)
            print(f"\nIntermediate PDF saved to {batch_output_path} (pages {page_batch_start+1} to {page_batch_end})")

    # Save the final document
    #translated_doc.save(output_pdf_path)
    compressed_output = f"{base_name}_compressed{ext}"
    translated_doc.save(compressed_output, garbage=4, deflate=True, clean=True)
    print(f"\nFinal translated PDF saved to {output_pdf_path}")
    print(f"Compressed version saved to {compressed_output}")



def bulck_translate_files(folder_path, result_path):
    if isinstance(folder_path,str):
        pdf_files = os.listdir(folder_path)
        pdf_files = [os.path.join(folder_path,file) for file in pdf_files if str(file).endswith('.pdf')]
    else:
        pdf_files = folder_path
    for file in pdf_files:
        result_file = os.path.join(result_path,file.split('/')[-1])
        translate_pdf_text_precise(file,result_file)
        break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Translate PDF text with precise positioning')
    parser.add_argument('--input', type=str, default=r"C:\Users\161070\Downloads\pdf_file - Copy.pdf",
                        help='Path to input PDF file')
    parser.add_argument('--output', type=str, default="translated_text_preserved_deep.pdf",
                        help='Path to save translated PDF')
    parser.add_argument('--translator', type=str, choices=['local', 'google', 'deep'], default='deep',
                        help='Translator to use: local (mBART model), google (Google Translate API), or deep (Deep Translator)')
    parser.add_argument('--source', type=str, default='zh-CN',
                        help='Source language code (default: zh-CN)')
    parser.add_argument('--target', type=str, default='en',
                        help='Target language code (default: en)')

    args = parser.parse_args()

    print(f"Using {args.translator} translator")
    df = pd.read_csv('Data\Copy of chinees_only.csv')
    links = df['SE_URL'].to_list()
    #translate_pdf_text_precise(args.input, args.output, args.source, args.target, args.translator)
    bulck_translate_files(links,'Results')
