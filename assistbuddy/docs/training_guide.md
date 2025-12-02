# Training Guide - AssistBuddy

## Prerequisites

### Hardware Requirements
- **For Google Colab (Recommended)**:
  - GPU: T4 (free tier) or A100/V100 (Pro)
  - RAM: 12GB+ (standard Colab)
  - Storage: 15GB+ on Google Drive for checkpoints

- **For Local Training**:
  - GPU: NVIDIA RTX 3090/4090 or better (24GB+ VRAM)
  - RAM: 32GB+ system RAM
  - Storage: 50GB+ SSD space

### Software Requirements
- Python 3.8+
- CUDA 11.7+ (for local GPU training)
- Google account (for Colab)

## Step 1: Prepare Data

### Option A: Use Synthetic Data (Recommended for Testing)

The system includes a dataset generator that creates synthetic training samples:

```python
from assistbuddy.data import DatasetGenerator

generator = DatasetGenerator(output_dir="./data")
metadata_path = generator.generate_dataset(
    num_invoices=100,
    num_cctv_frames=100
)
```

This generates:
- Invoice images with OCR text
- CCTV frames with activity labels (working, idle, absent)
- Ground truth summaries in both ADMIN and FRIEND styles

### Option B: Use Real Data

If you have real data, organize it as:

```
data/
├── images/           # Invoice scans, receipts, photos
├── pdfs/             # Documents, reports
├── videos/           # CCTV footage, monitoring videos
├── excel/            # Spreadsheets, data files
└── metadata.json     # Ground truth summaries
```

Format for `metadata.json`:

```json
[
  {
    "file": "invoice_001.png",
    "type": "image",
    "admin_summary": "TL;DR: ...",
    "friend_summary": "TL;DR: ...",
    "metadata": {...}
  }
]
```

## Step 2: Google Colab Setup

### 2.1 Open Notebook

1. Upload `colab_training_notebook.ipynb` to Google Drive
2. Open with Google Colab
3. Go to **Runtime → Change runtime type**
4. Select **GPU** (T4 for free, A100 for Pro)

### 2.2 Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

This allows saving checkpoints to your Drive.

### 2.3 Upload Code

**Option A - Upload Folder**:
1. Zip the `assistbuddy` folder
2. Upload to Colab
3. Unzip: `!unzip assistbuddy.zip`

**Option B - Clone from GitHub**:
```bash
!git clone https://github.com/yourusername/chatarchitect.git
!cd chatarchitect/assistbuddy
```

## Step 3: Install Dependencies

```bash
!pip install -q torch torchvision torchaudio
!pip install -q transformers accelerate bitsandbytes peft
!pip install -q opencv-python pytesseract pdfplumber
!pip install -q openai-whisper spacy gradio
!python -m spacy download en_core_web_sm
```

## Step 4: Configure Training

Edit `assistbuddy_config.py`:

```python
# Key settings to adjust
BATCH_SIZE = 4              # Reduce if OOM (out of memory)
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-5
MAX_EPOCHS = 5
CHECKPOINT_EVERY_N_STEPS = 500
```

**Memory Tips**:
- If you get OOM errors, reduce `BATCH_SIZE` to 2
- Increase `GRADIENT_ACCUMULATION_STEPS` to maintain effective batch size
- Use mixed precision training (enabled by default)

## Step 5: Run Training

Execute all cells in the Colab notebook sequentially:

### 5.1 Generate Data
```python
generator = DatasetGenerator()
generator.generate_dataset(num_invoices=100, num_cctv_frames=100)
```

### 5.2 Build Model
```python
from assistbuddy import AssistBuddyModel
model = AssistBuddyModel(...)
model = model.to('cuda')
```

### 5.3 Train
```python
for epoch in range(config['epochs']):
    train_epoch(model, dataloader, optimizer, scheduler, device, epoch)
```

**Training Time Estimates** (T4 GPU):
- 100 samples: ~30 minutes
- 500 samples: ~2 hours
- 1000 samples: ~4 hours
- 5000 samples: ~8 hours

### 5.4 Monitor Progress

Watch for:
- ✅ Loss decreasing steadily
- ✅ Checkpoints saving to Drive
- ⚠️ Loss stuck → reduce learning rate
- ⚠️ NaN loss → reduce learning rate, check data

## Step 6: Checkpointing

Checkpoints are automatically saved to:
```
/content/drive/MyDrive/assistbuddy_checkpoints/
checkpoint_epoch1_step500.pt
checkpoint_epoch1_final.pt
...
```

### Resume from Checkpoint

```python
checkpoint = torch.load('/content/drive/MyDrive/assistbuddy_checkpoints/checkpoint_epoch2_final.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
```

## Step 7: Evaluation

Test on validation samples:

```python
model.eval()
with torch.no_grad():
    for sample in val_dataloader:
        outputs = model(sample['images'], sample['text_inputs'], style='admin')
        generated = tokenizer.decode(outputs.logits.argmax(-1)[0])
        print(generated)
```

## Step 8: Save Final Model

```python
final_dir = "/content/drive/MyDrive/assistbuddy_final"
model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)
```

## Step 9: Deploy

### Option A: Gradio Demo

```python
import gradio as gr

def summarize(file, style):
    # Process file
    # Generate summary
    return summary

demo = gr.Interface(fn=summarize, ...)
demo.launch(share=True)
```

### Option B: API Server

```python
from fastapi import FastAPI, File, UploadFile
app = FastAPI()

@app.post("/summarize")
async def summarize_file(file: UploadFile, style: str):
    # Process and return summary
    return {"summary": "..."}
```

## Troubleshooting

### Out of Memory (OOM)
- Reduce `BATCH_SIZE` to 2 or 1
- Reduce `MAX_TOKENS` to 256
- Use gradient checkpointing
- Clear GPU cache: `torch.cuda.empty_cache()`

### Slow Training
- Ensure GPU is enabled (check with `torch.cuda.is_available()`)
- Use mixed precision (enabled by default)
- Increase batch size if memory allows

### Poor Quality Outputs
- Train longer (more epochs)
- Increase dataset size
- Adjust temperature (lower for ADMIN, higher for FRIEND)
- Fine-tune on real data

### Colab Disconnects
- Keep tab active (use browser extension to prevent sleep)
- Enable automatic reconnection
- Save checkpoints frequently
- Use Colab Pro for longer sessions

## Advanced: Custom Data

To train on your own data:

1. Create ground truth summaries for your files
2. Format as JSON (see Option B in Step 1)
3. Create custom DataLoader:

```python
class CustomDataset(Dataset):
    def __init__(self, metadata_path):
        with open(metadata_path) as f:
            self.samples = json.load(f)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        # Load file, process, return tensors
        return {
            'images': ...,
            'text_inputs': ...,
            'labels': ...,
            'style': ...
        }
```

## Next Steps

- Fine-tune on domain-specific data
- Add more file types (Word, PowerPoint)
- Improve activity recognition with better vision models
- Deploy to production with API
- Integrate with existing admin dashboards

For questions or issues, refer to the main README.md
