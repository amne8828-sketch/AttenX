# AssistBuddy - Quick Start Guide

## What is AssistBuddy?

AssistBuddy is a multimodal AI assistant that reads and summarizes:
- 📷 **Images** (invoices, receipts, photos with OCR)
- 📄 **PDFs** (documents, reports, scanned files)
- 📊 **Excel** files (with statistics and outlier detection)
- 🎥 **Videos** (CCTV footage, monitoring cameras)
- 🌐 **Webpages** (site analysis and state reports)

**Two Communication Styles:**
1. **ADMIN** - Professional, structured summaries
2. **FRIEND** - Casual WhatsApp-style Hinglish tone

**Special Features:**
- ✅ Human activity detection (camera monitoring: working/idle/absent)
- ✅ Automatic PII detection and redaction
- ✅ Provenance tracking (sources, confidence scores)
- ✅ Privacy-first design

## Installation

### Option 1: Google Colab (Recommended)

1. **Open the Training Notebook**
   ```
   Upload colab_training_notebook.ipynb to Google Drive
   Open with Google Colab
   ```

2. **Enable GPU**
   ```
   Runtime → Change runtime type → GPU (T4)
   ```

3. **Run All Cells**
   - Installs dependencies
   - Generates training data
   - Trains the model
   - Saves to Google Drive

### Option 2: Local Installation

```bash
# Clone or navigate to assistbuddy folder
cd chatarchitect/assistbuddy

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Generate training data
python -c "from data import DatasetGenerator; DatasetGenerator().generate_dataset(100, 100)"
```

## Quick Training

### Google Colab (6-8 hours)

```python
# In colab_training_notebook.ipynb
# Just run all cells sequentially!

# Checkpoints auto-save to:
# /content/drive/MyDrive/assistbuddy_checkpoints/
```

### Local (with GPU)

```python
from assistbuddy import AssistBuddyModel
from assistbuddy.data import DatasetGenerator

# Generate data
generator = DatasetGenerator()
generator.generate_dataset(num_invoices=100, num_cctv_frames=100)

# Train (see docs/training_guide.md for full code)
model = AssistBuddyModel(...)
train(model, dataloader, epochs=5)
```

## Usage Examples

### Example 1: Summarize Invoice

```python
from assistbuddy import AssistBuddyModel
from PIL import Image

model = AssistBuddyModel.load_pretrained("path/to/checkpoint")
image = Image.open("invoice_104.pdf")

# ADMIN style
admin_summary = model.summarize(image, style="admin")
print(admin_summary)

# Output:
# TL;DR: Invoice_104 shows pending payment of ₹1,24,500 due 10 Nov 2025. (Confidence 92)
# 
# Key details:
# - Invoice ID: 104, Issuer: Aakash Furnitures Pvt Ltd (invoice_104.png)
# - Amount due: ₹1,24,500 (GST incl). Due date: 2025-11-10.
# ...

# FRIEND style
friend_summary = model.summarize(image, style="friend")
print(friend_summary)

# Output:
# TL;DR: Boss, invoice_104 pending — ₹1,24,500 due by 10 Nov 😬 (Conf 92)
# 
# Key bits:
# - Total: ₹1,24,500 (with GST). Due: 10 Nov 2025.
# - Status: Payment pending, thoda urgent.
# ...
```

### Example 2: Camera Monitoring

```python
from assistbuddy.utils import VideoProcessor

processor = VideoProcessor()
processor.load_video("cctv_footage.mp4")

# Extract keyframes
keyframes = processor.extract_keyframes(num_frames=10)

# Summarize each frame
for frame in keyframes:
    summary = model.summarize(frame.frame, style="admin")
    print(f"@ {frame.timestamp}: {summary}")

# Output:
# @ 00:14:32: Person detected working at desk (Confidence 85%)
# @ 00:28:15: Workspace empty, person absent (Confidence 95%)
# @ 00:42:08: Person on phone, not actively working (Confidence 78%)
```

### Example 3: Excel Analysis

```python
from assistbuddy.utils import ExcelParser

parser = ExcelParser()
parser.load_excel("sales_data.xlsx")

# Get text representation
excel_text = parser.get_text_representation()

# Summarize
summary = model.summarize(excel_text, style="admin", file_type="excel")
print(summary)

# Output:
# TL;DR: Sales spreadsheet shows 8% drop this week with suspicious outliers in row 14. (Confidence 88)
# 
# Key details:
# - Rows: 150, Columns: 12, Missing values: 8 (sales_data.xlsx)
# - Outliers detected: Row 14 (₹5,00,000 - exceeds normal range by 3.2σ)
# ...
```

### Example 4: PDF Document

```python
from assistbuddy.utils import PDFParser

parser = PDFParser(use_ocr=True)
result = parser.extract_full_document("report.pdf")

summary = model.summarize(result['text'], style="admin", file_type="pdf")
print(summary)
```

## JSON Output Format

Enable structured JSON output:

```python
summary_json = model.summarize(file, style="admin", output_format="json")

# Returns:
{
  "tldr": "Invoice_104 shows pending payment...",
  "style": "ADMIN",
  "confidence_overall": 92,
  "key_details": [
    {"text": "Amount: ₹1,24,500", "confidence": 95, "source": "invoice_104.png:OCR"}
  ],
  "actions": [
    {
      "text": "Contact accounts to confirm payment",
      "priority": "High",
      "why": "Payment overdue",
      "confidence": 90
    }
  ],
  "provenance": [
    {"file": "invoice_104.png", "type": "image", "method": "OCR", "page_or_ts": ""}
  ],
  "notes": "PII detected and redacted: 1 phone number"
}
```

## Privacy Features

### Automatic PII Redaction

```python
from assistbuddy.privacy import PIIDetector, PIIRedactor

text = "Contact John at john@example.com or +91 9876543210"

detector = PIIDetector()
redactor = PIIRedactor(detector)

redacted, detections = redactor.redact(text)
print(redacted)
# Output: "Contact [REDACTED_NAME] at [REDACTED_EMAIL] or [REDACTED_PHONE]"

warning = redactor.generate_privacy_warning(detections)
print(warning)
# Output: "⚠️ Privacy Warning: Detected 1 email, 1 phone_india, 1 name_person..."
```

## Activity Recognition for Camera Monitoring

AssistBuddy can analyze camera feeds to detect if people are:
- ✅ **Working** - Actively engaged in tasks
- ⚠️ **Idle** - Present but not working (on phone, chatting)
- 🚫 **Absent** - Workspace empty
- 🚶 **Transit** - Walking through area
- ❓ **Unclear** - Motion blur, poor quality

Example admin query:
```
"Summarize today's camera view from CAM-01"
```

Response:
```
TL;DR: 8 hours monitored. 6h active work detected, 1.5h idle periods, 0.5h absent.

Timeline:
- 09:00-12:30: Person working at desk (Conf 85-92%)
- 12:30-13:00: Absent (lunch break)
- 13:00-15:45: Person working (Conf 88-94%)
- 15:45-16:15: Idle on phone (Conf 76%)
- 16:15-17:00: Person working (Conf 90%)

Actions:
- Review 15:45-16:15 idle period if outside break time
```

## Deployment

### Gradio Demo

```python
import gradio as gr
from assistbuddy import AssistBuddyModel

model = AssistBuddyModel.load_pretrained("checkpoint")

def summarize_file(file, style):
    summary = model.summarize(file, style=style.lower())
    return summary

demo = gr.Interface(
    fn=summarize_file,
    inputs=[
        gr.File(label="Upload File"),
        gr.Radio(["ADMIN", "FRIEND"], value="ADMIN")
    ],
    outputs=gr.Textbox(label="Summary", lines=15),
    title="AssistBuddy - AI Summarizer"
)

demo.launch()
```

### FastAPI Server

```python
from fastapi import FastAPI, File, UploadFile
from assistbuddy import AssistBuddyModel

app = FastAPI()
model = AssistBuddyModel.load_pretrained("checkpoint")

@app.post("/summarize")
async def summarize(file: UploadFile, style: str = "admin"):
    content = await file.read()
    summary = model.summarize(content, style=style)
    return {"summary": summary}

# Run with: uvicorn app:app --host 0.0.0.0 --port 8000
```

## File Structure

```
assistbuddy/
├── model/                    # Neural network architecture
│   ├── assistbuddy_model.py  # Main model
│   ├── vision_encoder.py     # CLIP encoder
│   ├── text_encoder.py       # BERT encoder
│   ├── audio_encoder.py      # Whisper encoder
│   ├── multimodal_fusion.py  # Fusion layer
│   └── decoder.py            # GPT-2 decoder
├── privacy/                  # PII and compliance
│   ├── pii_detector.py
│   ├── redactor.py
│   └── provenance_tracker.py
├── utils/                    # File processing
│   ├── ocr_engine.py
│   ├── pdf_parser.py
│   ├── excel_parser.py
│   ├── video_processor.py
│   └── web_scraper.py
├── data/                     # Dataset generation
│   └── dataset_generator.py
├── inference/                # Style templates
│   └── style_prompts.py
├── docs/                     # Documentation
│   └── training_guide.md
├── colab_training_notebook.ipynb  # Google Colab notebook
├── assistbuddy_config.py          # Configuration
├── requirements.txt
└── README.md
```

## Next Steps

1. **Train on Your Data**
   - Replace synthetic data with real invoices, CCTV footage
   - Fine-tune for your specific domain

2. **Deploy to Production**
   - Set up API server
   - Integrate with admin dashboard
   - Configure role-based access

3. **Extend Functionality**
   - Add more file types (Word, PowerPoint)
   - Improve activity recognition
   - Add multilingual support

## Support

- 📖 Full training guide: `docs/training_guide.md`
- 📝 Main README: `README.md`
- 💻 Google Colab notebook: `colab_training_notebook.ipynb`

## License

MIT License
