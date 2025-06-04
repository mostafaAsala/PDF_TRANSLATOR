import io
import argparse
import os
import time
import tempfile
import shutil
import requests
from urllib.parse import urlparse
import pandas as pd
import traceback
import fitz  # PyMuPDF
import unicodedata
from PIL import Image
from googletrans import Translator as GoogleTranslator
from deep_translator import GoogleTranslator as DeepGoogleTranslator
# Load model directly
print("loading packages")
import pandas as pd
src = "zh"  # source language
trg = "en"  # target language
print("loading model")
model_name = f"Helsinki-NLP/opus-mt-{src}-{trg}"
font_path ='ARIAL.TTF'
text_font = fitz.Font(fontfile=font_path) 
# Initialize Google Translator
google_translator = GoogleTranslator()

# Initialize Deep Translator
deep_translator = None

# Only load the model if needed (will be loaded on demand)
tokenizer = None
model = None

def load_local_model():
    from transformers import AutoTokenizer, MarianMTModel, AutoModelForSeq2SeqLM
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


def get_page_pixmap_from_xref(doc, xref, matrix=None):
    # Render a Form XObject into an image
    xobj = fitz.xref_object(doc, xref, compressed=True)
    form = fitz.open("pdf", xobj.encode("utf-8"))  # fake PDF from the Form
    page = form[0]
    matrix = matrix or fitz.Matrix(1, 1)
    pix = page.get_pixmap(matrix=matrix, alpha=True)
    return pix


# Monkey patch (optional, for reuse)
fitz.Document.get_page_pixmap_from_xref = get_page_pixmap_from_xref


def extract_image_or_form_as_image(doc, xref, bbox, matrix=None):
    obj = doc.xref_object(xref, compressed=False)
    
    # --- Case 1: Regular image ---
    if "/Subtype /Image" in obj:
        return extract_image_with_transparency(doc, xref)

    # --- Case 2: Form XObject ---
    if "/Subtype /Form" in obj:
        # Create a temporary page to render the form
        # Define a matrix for scaling if needed
        form_rect = fitz.Rect(bbox)
        matrix = matrix or fitz.Matrix(1, 1)  # scale = 1

        pix = doc.get_page_pixmap_from_xref(xref, matrix=matrix)
        image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGBA")
        return image

    # --- Unknown subtype ---
    raise ValueError(f"Unsupported XObject subtype in xref {xref}")

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

    # Step 4: Resize alpha mask if needed
    if img.size != smask.size:
        smask = smask.resize(img.size, resample=Image.BILINEAR)
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

def contains_chinese(text):
    """
    Check if the given text contains Chinese characters.
    
    Args:
        text: Text to check for Chinese characters
        
    Returns:
        True if the text contains Chinese characters, False otherwise
    """
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            return False
            
    # Unicode ranges for Chinese characters
    # CJK Unified Ideographs (Common): U+4E00 - U+9FFF
    # CJK Unified Ideographs Extension A: U+3400 - U+4DBF
    # CJK Unified Ideographs Extension B: U+20000 - U+2A6DF
    # CJK Unified Ideographs Extension C: U+2A700 - U+2B73F
    # CJK Unified Ideographs Extension D: U+2B740 - U+2B81F
    # CJK Unified Ideographs Extension E: U+2B820 - U+2CEAF
    # CJK Unified Ideographs Extension F: U+2CEB0 - U+2EBEF
    
    for char in text:
        # Check if the character is in the main CJK Unified Ideographs range or Extension A
        if ('一' <= char <= '鿿') or ('㐀' <= char <= '䶿'):
            return True
        
        # Check for characters in higher Unicode planes (Extension B-F)
        # These are represented as surrogate pairs in Python strings
        cp = ord(char)
        if 0x20000 <= cp <= 0x2EBEF:
            return True
            
    return False

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
    try:
        # Open the original document
        doc = fitz.open(input_pdf_path)
        total_pages = len(doc)

        # Create a new empty document for the translation
        translated_doc = fitz.open()

        # Process pages in batches of 10
        save_interval = 10

        # Get base name and extension for intermediate saves
        
        
        base_name = os.path.basename(output_pdf_path)
        base_name, ext = os.path.splitext(base_name)
        # Create a directory for batch files
        batch_dir = os.path.join(os.path.dirname(output_pdf_path), 'batch_files')
        os.makedirs(batch_dir, exist_ok=True)

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
                                text = span["text"]#.strip()
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
                    # Filter out only the Chinese text for translation
                    chinese_texts = []
                    chinese_indices = []
                    
                    # Identify Chinese text and their indices
                    for idx, text in enumerate(page_text_to_translate):
                        if contains_chinese(text):
                            chinese_texts.append(text)
                            chinese_indices.append(idx)
                    
                    # Initialize translations list with the original texts
                    page_translations = page_text_to_translate.copy()
                    
                    if chinese_texts:
                        print(f"Translating {len(chinese_texts)} Chinese text elements from page {page_number+1} using {translator_type} translator...")
                        
                        # Process Chinese texts in batches to avoid memory issues
                        batch_size = 50  # Adjust based on your model's capacity
                        chinese_translations = []
                        
                        for i in range(0, len(chinese_texts), batch_size):
                            batch_texts = chinese_texts[i:i+batch_size]
                            try:
                                # Choose translation method based on user selection
                                if translator_type.lower() == 'google':
                                    batch_translations = batch_translate_with_google(batch_texts, source_lang, target_lang)
                                elif translator_type.lower() == 'deep':
                                    batch_translations = batch_translate_with_deep(batch_texts, source_lang, target_lang)
                                else:  # Default to local model
                                    batch_translations = batch_translate_with_local_model(batch_texts)

                                chinese_translations.extend(batch_translations)
                                print(f"Translated batch {i//batch_size + 1}/{(len(chinese_texts) + batch_size - 1)//batch_size}")

                            except Exception as e:
                                print(f"Batch translation error: {traceback.format_exc()}")
                                # If batch fails, add original text as fallback
                                chinese_translations.extend(batch_texts[len(chinese_translations) - i:])
                        
                        # Replace the Chinese texts with their translations in the original order
                        for idx, trans_idx in enumerate(chinese_indices):
                            if idx < len(chinese_translations):
                                page_translations[trans_idx] = chinese_translations[idx]
                        
                        print(f"Successfully translated {len(chinese_texts)} Chinese text elements")
                    else:
                        print(f"No Chinese text to translate on page {page_number+1}")
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
                        try:
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
                                quad = item[1]
                                if len(item) >= 3:
                                    p1, p2 = item[1], item[2]
                                    shape.draw_line(p1, p2)
                                else:
                                    
                                    shape.draw_quad(quad)
                                    # shape.draw_line(quad.p1, quad.p2)  # Or draw the full quad as 4 lines
                                    # shape.draw_line(quad.p2, quad.p3)
                                    # shape.draw_line(quad.p3, quad.p4)
                                    # shape.draw_line(quad.p4, quad.p1)
                            elif item[0] == "c":  # curve
                                shape.draw_bezier(item[1], item[2], item[3], item[4])
                            else:
                                print(f"Unknown drawing item type: {item[0]}")
                        except Exception as e:
                            print(f"Error processing drawing item: {traceback.format_exc()}")
                    
                    try:
                        shape.finish(width=drawing['width'],
                                color=drawing['color'],
                                closePath=drawing['closePath'],
                                fill=drawing['fill'],
                                dashes=drawing['dashes'],
                                stroke_opacity=1 if drawing.get("stroke_opacity", 1) is None else drawing.get("stroke_opacity", 1))
                        shape.commit()
                    except Exception as e:
                        print(f"Error finishing shape: {traceback.format_exc()}")
                print("drawing2")
                # 2. Extract and copy all images
                image_list = page.get_image_info(hashes=False, xrefs=True)
                for idx, img_info in enumerate(image_list):
                    try:
                        xref = img_info['xref']

                        # Get the bounding box for the image
                        bbox = img_info['bbox']

                        image = extract_image_or_form_as_image(doc, xref, bbox)

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
                        if image.mode not in ("RGB", "RGBA"):
                            image = image.convert("RGBA")
                        # Insert the image using raw bytes (preserves alpha if present)
                        new_page.insert_image(bbox, 
                            stream=modified_img_bytes, 
                            keep_proportion=False,
                            width=image.width,
                            height=image.height,
                            xref=0,  # allow MuPDF to assign new xref
                            overlay=True  # insert over existing content if needed
                            )
                    except Exception as e:
                        print(f"Error processing image: {traceback.format_exc()}")
                print("image")
                # 3. Add translated text for this page
                if page_text_to_translate:
                    for idx, (text, meta) in enumerate(zip(page_text_to_translate, page_text_metadata)):
                        try:
                            if idx < len(page_translations):
                                # Check if the original text contains Chinese characters
                                if contains_chinese(text):
                                    # Use the translated text if the original contains Chinese
                                    display_text = page_translations[idx]
                                else:
                                    # Use the original text if it doesn't contain Chinese
                                    display_text = text
                    
                                # Get metadata
                                bbox = meta['bbox']
                                font_size = meta['font_size']
                                color_int = meta['color_int']
                                
                                # Process and insert the text
                                display_text_clean = display_text #sanitize_text(display_text)
                                try:
                                    text_width = fitz.get_text_length(display_text_clean, fontname="custom",  # standard font
                                                        fontfile="C:/Windows/Fonts/arial.ttf",fontsize=font_size)
                                except:
                                    text_width = fitz.get_text_length(display_text_clean, fontname='helv',fontsize=font_size)
                                bbox_width = bbox[2] - bbox[0]

                                scale_x = bbox_width / text_width if text_width > 0 else 1.0
                                if scale_x > 1:
                                    scale_x = 1
                                color_rgb = int_to_rgb(color_int)
                                pivot = fitz.Point(bbox[0], bbox[1])
                                mat = fitz.Matrix(scale_x, 1)
                                morph = (pivot, mat)
                                """display_text = display_text.replace("，", ",")
                                display_text = display_text.replace("。", ".")
                                display_text = display_text.replace("？", "?")
                                display_text = display_text.replace("！", "!")
                                display_text = display_text.replace("；", ";")
                                display_text = display_text.replace("：", ":")
                                display_text = display_text.replace("（", "(")
                                display_text = display_text.replace("）", ")")
                                display_text = display_text.replace("【", "[")
                                display_text = display_text.replace("】", "]")
                                display_text = display_text.replace("《", "<")
                                display_text = display_text.replace("》", ">")
                                display_text = display_text.replace("、", ",")
                                display_text = display_text.replace("、", ",")
                                display_text = display_text.replace("＝", "=")
                                display_text = display_text.replace("＋", "+")
                                display_text = display_text.replace("－", "-")
                                display_text = display_text.replace("＊", "*")
                                display_text = display_text.replace("／", "/")
                                display_text = display_text.replace("％", "%")
                                display_text = display_text.replace("％", "%")
                                display_text = display_text.replace("≤", "<=")
                                display_text = display_text.replace("≥", ">=")
                                display_text = display_text.replace("℃", "C")
                                display_text = display_text.replace("℉", "F")
                                display_text = display_text.replace("∆", "Δ")
                                display_text = display_text.replace("±", "±")"""
                                def unicode_normalize(text):
                                    return unicodedata.normalize("NFKC", text)

                                display_text = unicode_normalize(display_text)
                                new_page.insert_text(
                                    (bbox[0], bbox[1]+10),
                                    display_text,
                                    fontsize=font_size ,
                                    fontname="arial",  # standard font
                                    fontfile="C:/Windows/Fonts/arial.ttf",
                                    color=color_rgb,
                                    morph=morph,
                                )
                        except Exception as e:
                            print(f"Error inserting text: {traceback.format_exc()}")
                print("text")
            print("Saving batch file")
            try:
                # Save the document after each batch of pages
                if page_batch_end >= page_batch_start + 1:  # Only save if at least one page was processed
                    # Create a filename for the batch file
                    batch_filename = f"{base_name}_pages_{1}_to_{page_batch_end}{ext}"
                
                    # Full path in the batch_files directory
                    batch_output_path = os.path.join(batch_dir, batch_filename)
                    # Create a new PDF document
                    copied_doc = fitz.open()

                    # Insert all pages from the original document
                    copied_doc.insert_pdf(translated_doc)
                    # Save the batch file
                    copied_doc.save(batch_output_path, garbage=4, deflate=True, clean=True)
                    print(f"\nIntermediate PDF saved to {batch_output_path} (pages {page_batch_start+1} to {page_batch_end})")
            except Exception as e:
                print(f"Error saving batch file: {traceback.format_exc()}")
        # Save the final document
        # Get the base name of the input file for the output filename
        input_base_name = os.path.basename(input_pdf_path)
        input_base_name, _ = os.path.splitext(input_base_name)
        
        # Create the output filename with the input file's base name
        output_filename = f"{input_base_name}_translated{ext}"
        
        # Get the output directory from the output_pdf_path
        output_dir = os.path.dirname(output_pdf_path)
        
        # Create the full output path in the specified output directory
        output_file_path = os.path.join(output_dir, output_filename)
        
        print("Saving final translated file...........")

        try:
        # Save the translated document with the input file's base name
            translated_doc.save(output_file_path, garbage=4, deflate=True, clean=True)
            print(f"\nFinal translated PDF saved to {output_file_path}")
        except Exception as e:
            print(f"Error saving final translated file: {traceback.format_exc()}")
            translated_doc.save(output_file_path)
            print(f"\nFinal translated PDF saved to {output_file_path}")
        # Return the path to the final translated file
        return output_file_path
    except Exception as e:
        print(f"Error saving final translated file: {traceback.format_exc()}")
        return None

def is_url(path):
    """Check if the given path is a URL."""
    try:
        result = urlparse(path)
        return all([result.scheme, result.netloc])
    except:
        return False

def download_pdf_from_url(url, temp_dir):
    """Download a PDF from a URL to a temporary directory.
    
    Args:
        url: URL of the PDF to download
        temp_dir: Temporary directory to save the downloaded PDF
        
    Returns:
        Path to the downloaded PDF file or None if download failed
    """
    try:
        print(f"Downloading PDF from {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Get the filename from the URL or use a default name
        filename = os.path.basename(urlparse(url).path)
        if not filename or not filename.lower().endswith('.pdf'):
            filename = f"downloaded_{int(time.time())}.pdf"
        
        # Save the file to the temporary directory
        file_path = os.path.join(temp_dir, filename)
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"PDF downloaded to {file_path}")
        return file_path
    except Exception as e:
        print(f"Error downloading PDF from {url}: {e}")
        return None

def read_links_from_csv(csv_path, url_column):
    """Read links from a CSV file.
    
    Args:
        csv_path: Path to the CSV file
        url_column: Name of the column containing URLs
        
    Returns:
        List of URLs
    """
    try:
        df = pd.read_csv(csv_path)
        if url_column in df.columns:
            links = df[url_column].tolist()
            # Filter out NaN values and empty strings
            links = [link for link in links if isinstance(link, str) and link.strip()]
            return links
        else:
            print(f"Column '{url_column}' not found in CSV file. Available columns: {df.columns.tolist()}")
            return []
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return []

def bulck_translate_files(input_source, result_path, source_lang='auto', target_lang='en', translator_type='deep'):
    """Translate multiple PDF files.
    
    Args:
        input_source: Path to folder with PDFs, list of PDF paths, or path to CSV file with links
        result_path: Path to save translated PDFs
        source_lang: Source language code
        target_lang: Target language code
        translator_type: Type of translator to use
    """
    # Create result directory if it doesn't exist
    os.makedirs(result_path, exist_ok=True)
    
    # Create a temporary directory for downloaded files
    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory: {temp_dir}")
    
    # Initialize a list to store translation log data
    translation_log = []
    
    try:
        # Case 1: Input is a CSV file
        if isinstance(input_source, str) and input_source.lower().endswith('.csv'):
            print(f"Reading links from CSV file: {input_source}")
            # Assume the column name is 'SE_URL' by default, can be parameterized if needed
            links = read_links_from_csv(input_source, 'SE_URL')
            pdf_files = []
            
            # Process each link
            for link in links:
                try:
                    if is_url(link):
                        # Download the PDF if it's a URL
                        pdf_path = download_pdf_from_url(link, temp_dir)
                        if pdf_path:
                            pdf_files.append(pdf_path)
                            # Store the original link and the downloaded path for logging
                            translation_log.append({"input_source": link, "input_path": pdf_path, "output_path": None})
                    elif os.path.exists(link) and link.lower().endswith('.pdf'):
                        # It's a local file path
                        pdf_files.append(link)
                        # Store the file path for logging
                        translation_log.append({"input_source": link, "input_path": link, "output_path": None})
                    else:
                        print(f"Invalid or non-existent PDF path: {link}")
                except Exception as e:
                    print(f"Error processing link {link}: {e}")
        
        # Case 2: Input is a directory
        elif isinstance(input_source, str) and os.path.isdir(input_source):
            print(f"Scanning directory for PDF files: {input_source}")
            pdf_files = []
            for file in os.listdir(input_source):
                if file.lower().endswith('.pdf'):
                    file_path = os.path.join(input_source, file)
                    pdf_files.append(file_path)
                    # Store the file path for logging
                    translation_log.append({"input_source": input_source, "input_path": file_path, "output_path": None})
        
        # Case 3: Input is a list of paths or URLs
        elif isinstance(input_source, list):
            print(f"Processing list of {len(input_source)} paths/URLs")
            pdf_files = []
            for item in input_source:
                try:
                    if is_url(item):
                        # Download the PDF if it's a URL
                        pdf_path = download_pdf_from_url(item, temp_dir)
                        if pdf_path:
                            pdf_files.append(pdf_path)
                            # Store the original link and the downloaded path for logging
                            translation_log.append({"input_source": item, "input_path": pdf_path, "output_path": None})
                    elif os.path.exists(item) and item.lower().endswith('.pdf'):
                        # It's a local file path
                        pdf_files.append(item)
                        # Store the file path for logging
                        translation_log.append({"input_source": item, "input_path": item, "output_path": None})
                    else:
                        print(f"Invalid or non-existent PDF path: {item}")
                except Exception as e:
                    print(f"Error processing item {item}: {e}")
        
        else:
            print(f"Unsupported input source type: {type(input_source)}")
            return
        
        # Process each PDF file
        print(f"Found {len(pdf_files)} PDF files to translate")
        for i, file in enumerate(pdf_files):
            try:
                # Get the filename for the output based on input filename
                input_filename = os.path.basename(file)
                input_base_name, ext = os.path.splitext(input_filename)
                result_file = os.path.join(result_path, f"{input_base_name}_translated{ext}")
                
                print(f"[{i+1}/{len(pdf_files)}] Translating: {input_filename}")
                output_path = translate_pdf_text_precise(file, result_file, source_lang, target_lang, translator_type)
                print(f"Translated PDF saved to: {output_path}")
                
                # Update the log with the output path
                for log_entry in translation_log:
                    if log_entry["input_path"] == file:
                        log_entry["output_path"] = output_path
                        break
            except Exception as e:
                print(f"Error translating {file}: {traceback.format_exc()}")
    
    finally:
        # Save the translation log to a CSV file
        if translation_log:
            log_file = os.path.join(result_path, f"translation_log_{int(time.time())}.csv")
            try:
                df = pd.DataFrame(translation_log)
                df.to_csv(log_file, index=False)
                print(f"Translation log saved to {log_file}")
            except Exception as e:
                print(f"Error saving translation log: {e}")
        
        # Clean up temporary directory
        print(f"Cleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Translate PDF text with precise positioning')
    parser.add_argument('--input', type=str, default=r"C:\Users\161070\Downloads\data1.pdf",
                        help='Path to input PDF file or CSV file with links or directory with PDFs')
    parser.add_argument('--output', type=str, default="Results",
                        help='Directory to save translated PDFs')
    parser.add_argument('--csv-column', type=str, default='SE_URL',
                        help='Column name in CSV file containing PDF links (default: SE_URL)')
    parser.add_argument('--translator', type=str, choices=['local', 'google', 'deep'], default='deep',
                        help='Translator to use: local (mBART model), google (Google Translate API), or deep (Deep Translator)')
    parser.add_argument('--source', type=str, default='zh-CN',
                        help='Source language code (default: zh-CN)')
    parser.add_argument('--target', type=str, default='en',
                        help='Target language code (default: en)')

    args = parser.parse_args()

    print(f"Using {args.translator} translator")
    
    # Determine input type and process accordingly
    input_path = args.input
    
    if input_path.lower().endswith('.csv'):
        print(f"Processing CSV file: {input_path}")
        # CSV file with links
        bulck_translate_files(input_path, args.output, args.source, args.target, args.translator)
    elif input_path.lower().endswith('.pdf'):
        print(f"Processing single PDF file: {input_path}")
        # Single PDF file
        os.makedirs(args.output, exist_ok=True)
        output_file = os.path.join(args.output, f"translated_{os.path.basename(input_path)}")
        translate_pdf_text_precise(input_path, output_file, args.source, args.target, args.translator)
    elif os.path.isdir(input_path):
        print(f"Processing directory of PDFs: {input_path}")
        # Directory with PDFs
        bulck_translate_files(input_path, args.output, args.source, args.target, args.translator)
    else:
        # Try to interpret as a URL
        if is_url(input_path) and input_path.lower().endswith('.pdf'):
            print(f"Processing PDF from URL: {input_path}")
            # Create a temporary directory
            temp_dir = tempfile.mkdtemp()
            try:
                # Download the PDF
                pdf_path = download_pdf_from_url(input_path, temp_dir)
                if pdf_path:
                    # Translate the PDF
                    os.makedirs(args.output, exist_ok=True)
                    output_file = os.path.join(args.output, f"translated_{os.path.basename(pdf_path)}")
                    translate_pdf_text_precise(pdf_path, output_file, args.source, args.target, args.translator)
            finally:
                # Clean up temporary directory
                shutil.rmtree(temp_dir, ignore_errors=True)
        else:
            print(f"Error: Input '{input_path}' is not a valid PDF file, CSV file, directory, or PDF URL.")
