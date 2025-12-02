# AssistBuddy - Multimodal LLM Training System

**AssistBuddy** is an admin-facing multimodal AI assistant that reads and summarizes images, PDFs, Word docs, Excel sheets, videos, and webpages with two communication styles:

1. **ADMIN STYLE**: Professional, structured summaries (TL;DR, Key Details, Actions & Risks)
2. **FRIEND STYLE**: Casual, WhatsApp-like Indian tone with Hinglish and emojis

## Features

✅ **Multimodal Input Processing**
- Images (OCR + Vision)
- PDFs (text extraction)
- Excel/Word (structured data parsing)
- Videos (frame extraction + audio transcription)
- Webpages (scraping + state analysis)

✅ **Smart Summarization**
- TL;DR (1-2 lines)
- Key Details (bulleted)
- Actions & Risks with priority
- Confidence scores (0-100)
- Provenance tracking (filename, page, timestamp)

✅ **Privacy & Compliance**
- Automatic PII detection
- Redaction of sensitive data
- Privacy warnings for admins

✅ **Dual Communication Styles**
- Admin: Professional, structured
- Friend: Casual, Hinglish, emoji-friendly

## Quick Start - Google Colab Training

1. **Upload to Google Colab**: Open `colab_training_notebook.ipynb` in Google Colab
2. **Connect GPU**: Runtime → Change runtime type → GPU (T4 or better)
3. **Run all cells**: The notebook will:
   - Install dependencies
   - Generate synthetic training data
   - Train the multimodal model
   - Save checkpoints to Google Drive
4. **Monitor training**: View loss curves and sample outputs

## Project Structure

```
assistbuddy/
├── colab_training_notebook.ipynb    # Main Google Colab notebook
├── model/
│   ├── assistbuddy_model.py         # Main model architecture
│   ├── vision_encoder.py            # Image/video encoder
│   ├── text_encoder.py              # Text encoder
│   ├── audio_encoder.py             # Audio encoder
│   ├── multimodal_fusion.py         # Fusion layer
│   └── decoder.py                   # Generation decoder
├── data/
│   ├── dataset_generator.py         # Synthetic data generator
│   ├── data_preprocessor.py         # Preprocessing pipeline
│   └── multimodal_dataset.py        # PyTorch dataset
├── privacy/
│   ├── pii_detector.py              # PII detection
│   ├── redactor.py                  # Redaction engine
│   └── provenance_tracker.py        # Source tracking
├── utils/
│   ├── ocr_engine.py                # OCR wrapper
│   ├── pdf_parser.py                # PDF extraction
│   ├── excel_parser.py              # Excel parsing
│   ├── video_processor.py           # Video processing
│   └── web_scraper.py               # Webpage scraping
└── inference/
    ├── summarizer.py                # Inference engine
    └── style_prompts.py             # Style templates
```

## Training on Google Colab

The training is optimized for Google Colab free tier (T4 GPU, 15GB GPU RAM):

- **Model size**: 1.3B parameters (fits in T4 memory)
- **Batch size**: 4 (with gradient accumulation)
- **Mixed precision**: FP16 for memory efficiency
- **Checkpointing**: Auto-saves to Google Drive every 500 steps
- **Training time**: ~6-8 hours for 5,000 samples

## Dataset

The system generates synthetic training data including:
- Invoices, receipts, forms (PDFs)
- Business spreadsheets (Excel)
- Reports (Word)
- Invoice ID: 104, Issuer: Aakash Furnitures Pvt Ltd (invoice_104.pdf p1)
- Amount due: ₹1,24,500 (GST incl). Due date: 2025-11-10.
- Line items: 10 office chairs @₹10,000 each, GST 18%.

Actions & Risks:
1. Action: Contact accounts to confirm payment (Priority: High)
2. Risk: Late fee possible after 2025-11-25. (Confidence 78)

Sources & Files:
- invoice_104.pdf — page 1 — parsed via OCR
```

### Friend Style
```
TL;DR: Face in frame is too blurry for reliable ID, boss. Thoda unclear. (Conf 34)

Key bits:
- Time: 00:14:32 from cctv.mp4 (extracted frame)
- Face: small + motion blur; most landmarks not readable
- Visible: dark jacket, white sneakers

What to do:
- Try nearby frames at +/− 2s
- Share with security but note "low confidence"

Sources:
- cctv_frame.jpg (frame @00:14:32) — visual inspection
```

## License

MIT License - See LICENSE file for details
