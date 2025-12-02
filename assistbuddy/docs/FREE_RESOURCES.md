# Free Resources Guide for AssistBuddy Training

## 🎯 Zero-Cost Training Strategy

Train AssistBuddy completely **FREE** using these resources!

## 🆓 Free GPU Platforms (100+ hours/month)

### 1. **Google Colab** ⭐ Best for Training
- **GPU**: Tesla T4 (15GB VRAM)
- **Free Hours**: 12-16 hours/day
- **RAM**: 12GB
- **Storage**: 100GB temporary
- **Setup**: Upload `colab_training_notebook.ipynb`
- **URL**: https://colab.research.google.com

**Pro Tips**:
- Don't close browser (keeps session alive)
- Save checkpoints to Google Drive every 30 min
- Use off-peak hours (midnight-6am) for longer sessions

### 2. **Kaggle** ⭐ High GPU Quota
- **GPU**: P100 (16GB) or T4 (15GB)
- **Free Hours**: 30 hours/week (resets weekly)
- **RAM**: 13GB
- **Storage**: 20GB persistent + 75GB tmp
- **Setup**: Create notebook → Settings → Accelerator → GPU
- **URL**: https://www.kaggle.com/code

**Advantages**:
- More consistent than Colab
- Can run 2 notebooks simultaneously
- Persistent storage for datasets

### 3. **Lightning AI Studios**
- **GPU**: T4
- **Free Hours**: 22 hours/month
- **RAM**: 16GB
- **Setup**: Create Studio → Hardware → GPU
- **URL**: https://lightning.ai

### 4. **Paperspace Gradient**
- **GPU**: Free-GPU tier
- **Free Hours**: 6 hours/session
- **Setup**: Create Notebook → Free-GPU
- **URL**: https://gradient.run

### **Optimal Rotation Strategy**:
```
Week 1: Google Colab daily (84 hours)
Week 2: Kaggle (30 hours) + Colab backup
Week 3: Lightning AI (22 hours) + Colab
Week 4: Mix all platforms

TOTAL: ~100 hours FREE GPU time/month ⚡
```

---

## 🤖 Free AI APIs

### 1. **Google Gemini API** ⭐ Best for Data Labeling
- **Model**: Gemini 1.5 Flash
- **Free Quota**: 15 requests/minute, 1500/day
- **Cost**: $0 (no credit card required)
- **Best For**: Caption generation, image analysis
- **Get API Key**: https://makersuite.google.com/app/apikey

**Usage**:
```python
import google.generativeai as genai

genai.configure(api_key='YOUR_KEY')
model = genai.GenerativeModel('gemini-1.5-flash')

# Generate captions for your images
response = model.generate_content(["Describe this invoice", image])
caption = response.text

# FREE: 1500 images/day = 45,000/month!
```

### 2. **Hugging Face Inference API**
- **Free Quota**: 1000 requests/month
- **Models**: BLIP-2, GPT-2, ViT, etc.
- **Cost**: $0
- **Get Token**: https://huggingface.co/settings/tokens

**Usage**:
```python
import requests

API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip2-opt-2.7b"
headers = {"Authorization": f"Bearer {YOUR_TOKEN}"}

with open("invoice.jpg", "rb") as f:
    data = f.read()

response = requests.post(API_URL, headers=headers, data=data)
caption = response.json()[0]['generated_text']
```

### 3. **Replicate API**
- **Free Credits**: $10 on signup (no credit card)
- **Models**: LLaVA, BLIP, CogVLM
- **Processing**: ~500-1000 images with free credits
- **URL**: https://replicate.com

### 4. **OpenRouter** ⭐ Free Multimodal Models
- **Free Models**: 
  - `google/gemini-flash-1.5` (free forever)
  - `meta-llama/llama-3.2-11b-vision` (free)
- **Setup**: https://openrouter.ai/keys
- **Unlimited**: Yes, for free models

**Usage**:
```python
import requests

response = requests.post(
    url="https://openrouter.ai/api/v1/chat/completions",
    headers={"Authorization": f"Bearer {OPENROUTER_KEY}"},
    json={
        "model": "google/gemini-flash-1.5",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image"},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        }]
    }
)
```

---

## 📊 Free Datasets

### For Invoice/Document Understanding
1. **SROIE** - 1K receipt images with annotations
   - URL: https://rrc.cvc.uab.es/?ch=13
   - Perfect for invoice training

2. **CORD** - 11K receipts
   - URL: https://github.com/clovaai/cord
   - High-quality annotations

3. **DocVQA** - 50K questions on 12K documents
   - URL: https://www.docvqa.org
   - Document Q&A dataset

### For General Vision-Language
1. **COCO Captions** - 828K images, 5 captions each
   - URL: https://cocodataset.org

2. **Conceptual Captions** - 3.3M image-text pairs
   - URL: https://ai.google.com/research/ConceptualCaptions

3. **LAION-400M** - 400M image-text pairs
   - URL: https://laion.ai/blog/laion-400-open-dataset/
   - Subset: Use LAION-5M for faster download

### For Video/CCTV
1. **ActivityNet** - 20K videos with activities
   - URL: http://activity-net.org
   - Perfect for activity recognition

2. **UCF101** - 13K videos, 101 actions
   - URL: https://www.crcv.ucf.edu/data/UCF101.php

---

## 💡 Optimization Strategies

### Strategy 1: Use APIs for Data Labeling
**Save 80% training time by using pre-labeled data**

```python
# Use Gemini API to label 1500 images/day for FREE
import google.generativeai as genai

genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
model = genai.GenerativeModel('gemini-1.5-flash')

labeled_data = []
for image_path in unlabeled_images[:1500]:  # Daily limit
    img = Image.open(image_path)
    
    # ADMIN style prompt
    admin_prompt = "Create a professional summary with TL;DR, key details, and action items"
    admin_summary = model.generate_content([admin_prompt, img]).text
    
    # FRIEND style prompt
    friend_prompt = "Create a casual Hinglish summary like chatting with a friend"
    friend_summary = model.generate_content([friend_prompt, img]).text
    
    labeled_data.append({
        'image': image_path,
        'admin_summary': admin_summary,
        'friend_summary': friend_summary
    })

# 1500 images/day × 30 days = 45,000 labeled samples FREE!
```

### Strategy 2: Rotate GPU Platforms
**Never run out of GPU time**

```python
# Monday-Wednesday: Google Colab (36 hours)
# Thursday-Friday: Kaggle (15 hours)
# Weekend: Lightning AI (10 hours)
# Backup: Paperspace (6 hours)

# Total: ~67 hours/week = ~268 hours/month FREE
```

### Strategy 3: Synthetic Data Generation
**Generate unlimited training data**

```python
from faker import Faker
from PIL import Image, ImageDraw, ImageFont
import albumentations as A

fake = Faker()
augmenter = A.Compose([
    A.Rotate(limit=10),
    A.GaussNoise(),
    A.ElasticTransform()
])

# Generate 10,000 invoices for FREE
for i in range(10000):
    # Create fake invoice
    invoice_data = {
        'company': fake.company(),
        'amount': fake.random_int(1000, 500000),
        'date': fake.date(),
        'items': [fake.catch_phrase() for _ in range(5)]
    }
    
    # Create image
    img = create_invoice_image(invoice_data)
    
    # Augment 5x
    for aug_idx in range(5):
        aug_img = augmenter(image=img)['image']
        save_image(aug_img, f"invoice_{i}_{aug_idx}.png")

# Result: 50,000 training samples at $0 cost!
```

### Strategy 4: Use Validation APIs
**Validate without GPU**

```python
# Use HF Inference API for validation (1000 free/month)
import requests

def validate_model(validation_set):
    API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip2-opt-2.7b"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    results = []
    for idx, image in enumerate(validation_set[:1000]):  # Free limit
        with open(image, "rb") as f:
            response = requests.post(API_URL, headers=headers, data=f.read())
        
        predicted = response.json()[0]['generated_text']
        results.append({'image': image, 'prediction': predicted})
    
    return results

# Saves GPU time for training instead of validation!
```

---

## 🚀 Complete FREE Training Pipeline

```python
# STEP 1: Generate synthetic data (FREE - unlimited)
from faker import Faker
generate_invoices(count=1000)
generate_cctv_frames(count=1000)

# STEP 2: Label with Gemini API (FREE - 1500/day)
import google.generativeai as genai
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
model = genai.GenerativeModel('gemini-1.5-flash')

for image in images:
    caption = model.generate_content(["Describe", image]).text
    save_label(image, caption)

# STEP 3: Train on Google Colab (FREE - 12-16 hrs/day)
# Upload colab_training_notebook.ipynb
# Runtime → Change runtime type → GPU (T4)
# Run all cells

# STEP 4: Validate with HF API (FREE - 1000/month)
validate_with_hf_api(validation_set)

# STEP 5: Deploy on HF Spaces (FREE tier available)
# Create Gradio app
# Push to https://huggingface.co/spaces/YOUR_NAME/assistbuddy

# TOTAL COST: $0 🎉
```

---

## 📦 Free Model Hosting

### 1. **Hugging Face Spaces** ⭐ Best Free Tier
- **CPU**: Free forever
- **GPU**: Free tier available (limited) or $9/month
- **Demo**: Gradio/Streamlit
- **URL**: https://huggingface.co/spaces

**Deploy**:
```bash
# Create app.py with your Gradio interface
git clone https://huggingface.co/spaces/YOUR_NAME/assistbuddy
cd assistbuddy
# Add your model files
git push
```

### 2. **Replicate**
- **Free**: $10 signup credits
- **Cost**: ~$0.001/inference after
- **Best for**: Production API

### 3. **Modal**
- **Free**: $30/month credits
- **Best for**: Serverless GPU inference

---

## 💰 Cost Comparison

| Resource | Free Tier | If You Pay |
|----------|-----------|------------|
| Google Colab | 12-16 hrs/day | Colab Pro: $10/month |
| Kaggle | 30 hrs/week | Always free |
| Gemini API | 1500/day | $0.35/1000 images |
| HF Inference | 1000/month | $9/month unlimited |
| Datasets | Unlimited | Free forever |
| Synthetic Data | Unlimited | Free forever |
| **MONTHLY TOTAL** | **$0** | **$19 (optional)** |

---

## ✅ Action Plan

### Week 1: Setup & Data Generation
- [ ] Sign up for Google Colab, Kaggle, Lightning AI
- [ ] Get API keys: Gemini, Hugging Face, Replicate
- [ ] Generate 1000 synthetic invoices (Faker)
- [ ] Label 1500 images with Gemini API
- **Cost: $0**

### Week 2: Training Phase 1
- [ ] Train on Google Colab (Stage 1: 12 hours)
- [ ] Train on Kaggle (Stage 2: 15 hours)
- [ ] Save checkpoints to Google Drive
- **Cost: $0**

### Week 3: Training Phase 2
- [ ] Continue on Colab (Stage 3: 24 hours)
- [ ] Lightning AI backup (10 hours)
- [ ] Validate with HF API (1000 samples)
- **Cost: $0**

### Week 4: Deployment
- [ ] Test model locally
- [ ] Deploy Gradio app to HF Spaces
- [ ] Share demo link
- **Cost: $0**

**Total Time**: 4 weeks  
**Total Cost**: **$0** 🎉

---

## 🎓 Pro Tips

1. **Maximize Colab uptime**: Don't close browser, use sleep prevention extensions
2. **Checkpoint frequently**: Save to Google Drive every 30 minutes
3. **Use off-peak hours**: Train at night for longer sessions
4. **Combine platforms**: Start on Colab, continue on Kaggle if disconnected
5. **Cache datasets**: Download once, reuse across platforms
6. **Label in batches**: Use Gemini API's 1500/day limit efficiently

---

## Resources Summary

**Total Free Resources Value**: **~$500/month** if you paid for equivalent services

- GPU Time: $200/month (100 hours × $2/hour)
- APIs: $150/month (Gemini + HF)
- Datasets: $100/month (if bought commercially)
- Hosting: $50/month (HF Spaces Pro equivalent)

**Your Cost**: **$0** ⚡

Start training now with zero investment!
