# 📄 PDF Translation WebApp

Translate scanned and digital PDF files from Chinese (or other languages) into English or other target languages — while preserving layout, formatting, and images — all through a powerful, web-based Flask interface.

---

## 🚀 Features

* 🧠 **Multi-engine translation support**: Google Translate, Deep Translator APIs, and local MarianMT/MBART models.
* 📁 **Precise layout preservation**: Keeps text positions, fonts, rotation, and embedded images in place.
* 🌐 **Multiple input options**:

  * Direct PDF file upload
  * URL to PDF
  * CSV file with multiple PDF links
* 👥 **User Authentication**: Register/login, upload history, and batch tracking.
* 📦 **Batch translation & download** with progress monitoring.
* 📸 **Image support**: Extracts and reinserts images from PDF using transparency-aware logic.

---

## 🗭 Project Structure

```text
PDF_TRANSLATOR/
├── app/
│   ├── utils/
│   │   ├── auth.py               # User registration/authentication
│   │   └── monitor.py            # Job and batch tracking utilities
│   ├── routes.py                 # Flask routes and web interface logic
│   └── pdf_translator_deep.py   # Core translation logic using multiple engines
├── templates/                    # HTML templates
├── static/                       # CSS/JS/images
├── uploads/                      # Uploaded and translated PDFs
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 🌐 Web Interface Preview

> ✅ *Live demo coming soon...*

### 🔐 Login Page

![Login](https://github.com/user-attachments/assets/7fde5e79-9af9-4fe8-8053-45e5ad3aa6e9)

### 📄 Upload Page

![Upload](https://github.com/user-attachments/assets/a5d17e54-fc87-459b-b5c1-da1f6643ad44)

### 📊 Job History

![History](https://github.com/user-attachments/assets/762d3c08-658d-44d1-a30e-915bed505519)

---

## 📊 Before vs After Translation Comparison

| Original PDF (Chinese)                   | Translated PDF (English)               |
| ---------------------------------------- | -------------------------------------- |
| ![Before](https://github.com/user-attachments/assets/19e175b4-2ade-4078-aa57-101978857fe6)
 | ![After](https://github.com/user-attachments/assets/45973dbe-2591-4688-9ebf-7a614df8f5b5)
 |

---

## ⚙️ How It Works

1. **User uploads** a file, URL, or CSV through the `/upload` route.
2. Flask backend stores the input and metadata.
3. Calls `translate_pdf_text_precise()` from `pdf_translator_deep.py`, which:

   * Extracts Chinese texts
   * Translates using selected engine
   * Rebuilds the page: text + images + layout
4. Output is saved in the `uploads/` folder.
5. User can download the output or view status in `/history`.

---

## 🛠️ Installation

```bash
git clone https://github.com/your-username/pdf-translator-webapp.git
cd pdf-translator-webapp
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 📌 Configuration

* Upload folder: set `UPLOAD_FOLDER` in Flask config
* Local model: download `Helsinki-NLP/opus-mt-zh-en` before using
* Font path: check `pdf_translator_deep.py` for font location

---

## 🔀 Run Locally

```bash
source .venv/bin/activate
export FLASK_APP=app/routes.py
flask run
```

Then open: [http://localhost:5000](http://localhost:5000)

---

## 📤 Upload Methods

| Method     | Input Format                                 | Notes                                        |
| ---------- | -------------------------------------------- | -------------------------------------------- |
| File       | PDF                                          | Uploaded directly via HTML form              |
| URL        | [https://.../file.pdf](https://.../file.pdf) | Downloaded and translated server-side        |
| CSV Upload | CSV with `url` column                        | Translates all PDFs in batch with monitoring |

---

## 🚙 Supported Translators

| Translator       | Notes                                               |
| ---------------- | --------------------------------------------------- |
| Google Translate | Fast, reliable, requires internet                   |
| Deep Translator  | Tries Linguee, MyMemory, Libre, Requests fallback   |
| Local Model      | Offline MarianMT (e.g., Helsinki-NLP/opus-mt-zh-en) |

---

## ✅ Example Usage

```python
from app.pdf_translator_deep import translate_pdf_text_precise

translated = translate_pdf_text_precise(
    input_pdf_path='data/input.pdf',
    output_pdf_path='Results/output.pdf',
    source_lang='zh-CN',
    target_lang='en',
    translator_type='deep'  # or 'local' / 'google'
)
```

---

## 📅 Output Location

* Translated PDFs: `/uploads/`
* Batches: `/uploads/batch_files/`
* Logs: CSV files with translation status per job

---

## 📘 To-Do

* [ ] Add OCR support for scanned PDFs
* [ ] Deploy to HuggingFace / Heroku
* [ ] Add language auto-detection
* [ ] Support for PDFMiner as alternate engine

---

## 🤝 Credits

* [PyMuPDF](https://pymupdf.readthedocs.io/)
* [Transformers](https://huggingface.co/transformers/)
* [Deep Translator](https://github.com/nidhaloff/deep-translator)

---

## 📄 Screenshot Folder (optional)

Create this directory to support preview images:

```bash
docs/images/
├── login.png
├── upload.png
├── history.png
├── sample_before.png
└── sample_after.png
```

---

## 📄 License

MIT License © \[Your Name or Org]
