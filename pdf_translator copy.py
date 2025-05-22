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
def translate_pdf_text_precise(input_pdf_path, output_pdf_path, source_lang='auto', target_lang='en'):
    # Open the original document
    doc = fitz.open(input_pdf_path)

    # Create a new empty document for the translation
    translated_doc = fitz.open()

    #translator = GoogleTranslator(source=source_lang, target=target_lang)
    
    page_num = 1
    for page_number, page in enumerate(doc):
        # Create a new blank page with the same dimensions
        new_page = translated_doc.new_page(width=page.rect.width, height=page.rect.height)
        print(page.rect.width,page.rect.height)
        # 1. First, extract and redraw all drawings (lines, shapes, etc.)



        # 3. Get text blocks
        blocks = page.get_text("dict")["blocks"]

        """# 4. Create white rectangles to cover original text areas
        for block in blocks:

            if block['type'] == 0:  # Text block
                for line in block["lines"]:
                    for span in line["spans"]:
                        bbox = span["bbox"]
                        # Create a white rectangle to cover original text
                        new_page.draw_rect(fitz.Rect(bbox), color=(1, 1, 1), fill=(1, 1, 1))
        """
        # 2. Extract and copy all images

        if True:
            drawings = page.get_drawings()
            for drawing in drawings:
                shape = new_page.new_shape()
                length = len(drawing["items"])
                print(drawing['fill'])
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
                        print(item[0])

                shape.finish(width=drawing['width'],
                             color=drawing['color'],
                             closePath = drawing['closePath'],
                             fill = drawing['fill'],
                             dashes = drawing['dashes'],
                             stroke_opacity= 1 if drawing.get("stroke_opacity", 1)==None else drawing.get("stroke_opacity", 1),

                             )
                shape.commit()

        image_list = page.get_image_info(hashes=False,xrefs=True)
        img_index = 1
        page_num += 1

        for idx,img_info in enumerate( image_list):
            xref = img_info['xref']
            image = extract_image_with_transparency(doc, xref)

            # Extract the image bytes and other info
            base_image = doc.extract_image(xref)

            img_bytes = base_image["image"]  # This keeps RGBA if original image has alpha
            bbox = img_info['bbox']

            image = image.resize((int(image.width * 0.8), int(image.height * 0.8)), resample=Image.LANCZOS)

            # Save modified image to bytes
            output = io.BytesIO()
            image.save(output, format="PNG",dpi=(72, 72), optimize=True)
            output.seek(0)
            modified_img_bytes = output.read()

            image.save(f'images/extracted_image{idx}.png', 'PNG')  # Save as PNG
            # Insert the image using raw bytes (preserves alpha if present)
            new_page.insert_image(bbox, stream=modified_img_bytes, keep_proportion=False)

        """text_data = []
        # Insert the image at the same location in the new PDF
        for block in blocks:
            if block['type'] == 0:  # Text block
                for line in block["lines"]:
                    for span in line["spans"]:
                        if span["text"].strip():
                            text_data.append(span["text"].strip())
        translations = translator.translate(text_data,src='zh-CN',dest=target_lang)
        
        index_text = 0"""
        # 5. Add translated text
        for block in blocks:

            if block['type'] == 0:  # Text block

                for line in block["lines"]:
                    # print(line)
                    for span in line["spans"]:

                        bbox = span["bbox"]

                        text = span["text"].strip()
                        font_size = span["size"]
                        color_int = span["color"]
                        text_font = span['font']
                        flags = span['flags']
                        if text:
                            try:
                                
                                """batch = tokenizer([text], return_tensors="pt")

                                generated_ids = model.generate(**batch)
                                translated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                                """
                                # translate Hindi to French
                                tokenizer.src_lang = "zh_CN"
                                encoded_hi = tokenizer(text, return_tensors="pt")
                                generated_tokens = model.generate(
                                    **encoded_hi,
                                    forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"]
                                )
                                translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

                                #translated_text = translator.translate(text,src='zh-CN',dest=target_lang)
                                #print(translated_text )
                                
                                #translated_text = translated_text.text
                                if '' in span["text"]:
                                    translated_text = '■  '
                            except Exception as e:
                                print(f"Translation error on page {page_number}: {e}")
                                translated_text = span["text"]

                            if translated_text != None:
                                text_font = 'helv'
                                print("translated_text",translated_text )
                                print("------------------------------------------------------")
                                #if ('Bold' in text_font) or (flags & 1):
                                #    text_font = 'helv'
                                    # translated_text = '<b>' + translated_text + '</b>'
                                # translated_text =  f'<div style="font-family:{text_font}; font-size:{font_size+4}px;">' + translated_text + '</div>'
                                def sanitize_text(text):
                                    if not isinstance(text, str):
                                        try:
                                            text = text.decode('utf-8', errors='ignore')
                                        except Exception:
                                            text = str(text)
                                    # Remove non-printable characters and control characters
                                    return ''.join(c for c in text if c.isprintable())
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
                pass
            elif block['type'] == 1: #image


                xref = block['image']
                """bbox = fitz.Rect(block["bbox"])
                print(bbox)
                print(bbox[2]-bbox[0], bbox[3]-bbox[1])
                shape = new_page.new_shape()
                shape.draw_rect(bbox)
                shape.finish(width=2 ,color = (1,1,1))
                shape.commit()"""
                """
                img_bytes = block["image"]
                img_index += 1

                # Insert the image at the same location in the new PDF
                new_page.insert_image(bbox, stream=img_bytes)"""

            else:
                print(block)

    translated_doc.save(output_pdf_path)
    translated_doc.save("compressed_output.pdf", garbage=4, deflate=True, clean=True)
    print(f"Translated PDF saved to {output_pdf_path}")


if __name__ == "__main__":
    input_pdf = r"C:\Users\161070\Downloads\o4ybafz0i7eafqw6abn61smnd1i676.pdf"
    #input_pdf = r"C:\Users\161070\Downloads\2409191731_hikvision-ds-ipc-b12hv3-ia-poe4mm_c41359959.pdf"
    output_pdf = "translated_text_preserved.pdf"
    translate_pdf_text_precise(input_pdf, output_pdf)
