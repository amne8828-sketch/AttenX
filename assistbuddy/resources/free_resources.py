import os
import requests
import base64
from typing import Optional, Dict, List
import json


class FreeAPIIntegration:
    
    
    def __init__(self):
        self.hf_token = os.getenv('HUGGINGFACE_TOKEN', '')
        self.gemini_key = os.getenv('GEMINI_API_KEY', '')
        self.replicate_token = os.getenv('REPLICATE_API_TOKEN', '')
        self.openrouter_key = os.getenv('OPENROUTER_API_KEY', '')
        self.together_key = os.getenv('TOGETHER_API_KEY', '')
    
    def huggingface_inference(
        self,
        image_path: str,
        model: str = "Salesforce/blip-image-captioning-large"
    ) -> str:
       
        API_URL = f"https://api-inference.huggingface.co/models/{model}"
        headers = {"Authorization": f"Bearer {self.hf_token}"}
        
        with open(image_path, "rb") as f:
            data = f.read()
        
        response = requests.post(API_URL, headers=headers, data=data)
        return response.json()[0]['generated_text']
    
    def gemini_vision_api(
        self,
        image_path: str,
        prompt: str = "Describe this image in detail"
    ) -> str:
        """
        Use Google Gemini API for vision tasks
        FREE: 15 requests/minute, 1500/day, no credit card
        
        Get API key: https://makersuite.google.com/app/apikey
        """
        import google.generativeai as genai
        
        genai.configure(api_key=self.gemini_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Load image
        from PIL import Image
        img = Image.open(image_path)
        
        response = model.generate_content([prompt, img])
        return response.text
    
    def replicate_api(
        self,
        image_path: str,
        model: str = "llava-v1.6-34b"
    ) -> str:
        """
        Use Replicate API for VLM inference
        FREE: $10 credits on signup (no card needed)
        
        Models:
        - llava-v1.6-34b: Best accuracy
        - llava-13b: Faster
        - cogvlm: Good for Chinese
        """
        import replicate
        
        # Convert image to base64
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()
        
        output = replicate.run(
            f"yorickvp/{model}",
            input={
                "image": f"data:image/png;base64,{image_data}",
                "prompt": "Describe this image"
            }
        )
        return output
    
    def openrouter_free_models(
        self,
        prompt: str,
        image_url: Optional[str] = None
    ) -> str:
        """
        Use OpenRouter for free multimodal models
        FREE: Several free models available
        
        Free models:
        - google/gemini-flash-1.5 (free)
        - meta-llama/llama-3.2-11b-vision (free)
        """
        url = "https://openrouter.ai/api/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.openrouter_key}",
            "Content-Type": "application/json"
        }
        
        messages = [{"role": "user", "content": prompt}]
        
        if image_url:
            messages[0]["content"] = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        
        data = {
            "model": "google/gemini-flash-1.5",  # Free model
            "messages": messages
        }
        
        response = requests.post(url, headers=headers, json=data)
        return response.json()['choices'][0]['message']['content']


class FreeGPUResources:
    """
    Free GPU resources for training
    """
    
    @staticmethod
    def get_available_platforms():
        """
        List of free GPU platforms
        """
        return {
            'google_colab': {
                'gpu': 'T4 (15GB)',
                'free_hours': '12-16 hours/day',
                'upgrade': 'Colab Pro ($10/month) for A100',
                'url': 'https://colab.research.google.com',
                'setup': 'Click notebook.ipynb → Open with Colab'
            },
            'kaggle': {
                'gpu': 'P100 (16GB) or T4 (15GB)',
                'free_hours': '30 hours/week',
                'upgrade': 'Free forever',
                'url': 'https://www.kaggle.com/code',
                'setup': 'Create notebook → Enable GPU accelerator'
            },
            'lightning_ai': {
                'gpu': 'T4',
                'free_hours': '22 hours/month',
                'upgrade': 'Pay-as-you-go',
                'url': 'https://lightning.ai',
                'setup': 'Create Studio → Select GPU'
            },
            'paperspace_gradient': {
                'gpu': 'Free-GPU',
                'free_hours': '6 hours runtime',
                'upgrade': 'Growth ($8/month)',
                'url': 'https://gradient.run',
                'setup': 'Create notebook → Select free GPU'
            },
            'huggingface_spaces': {
                'gpu': 'T4 or A10G',
                'free_hours': 'Limited free tier',
                'upgrade': 'Pro ($9/month)',
                'url': 'https://huggingface.co/spaces',
                'setup': 'Can run training in Spaces'
            }
        }
    
    @staticmethod
    def best_strategy():
        """
        Optimal strategy for free GPU usage
        """
        return {
            'monday_wednesday': 'Google Colab (12-16 hrs)',
            'thursday_friday': 'Kaggle (15 hrs)',
            'weekend': 'Lightning AI (10 hrs)',
            'backup': 'Paperspace Gradient (6 hrs)',
            'total_free_hours': '~50 hours/week',
            'cost': '$0'
        }


class FreeDatasetResources:
    
    @staticmethod
    def get_free_datasets():
        """
        List of free multimodal datasets
        """
        return {
            'image_caption': {
                'coco_captions': {
                    'size': '828K images, 5 captions each',
                    'url': 'https://cocodataset.org',
                    'use': 'General image understanding'
                },
                'conceptual_captions': {
                    'size': '3.3M image-text pairs',
                    'url': 'https://ai.google.com/research/ConceptualCaptions',
                    'use': 'Web-scale captions'
                },
                'laion_400m': {
                    'size': '400M image-text pairs',
                    'url': 'https://laion.ai/blog/laion-400-open-dataset/',
                    'use': 'Large-scale pre-training'
                }
            },
            'document_understanding': {
                'docvqa': {
                    'size': '50K questions on 12K documents',
                    'url': 'https://www.docvqa.org',
                    'use': 'Document Q&A (invoices, forms)'
                },
                'sroie': {
                    'size': '1K receipts with annotations',
                    'url': 'https://rrc.cvc.uab.es/?ch=13',
                    'use': 'Receipt OCR and extraction'
                },
                'cord': {
                    'size': '11K receipts',
                    'url': 'https://github.com/clovaai/cord',
                    'use': 'Receipt understanding'
                }
            },
            'video_understanding': {
                'msrvtt': {
                    'size': '10K videos, 200K captions',
                    'url': 'https://www.microsoft.com/en-us/research',
                    'use': 'Video captioning'
                },
                'activitynet': {
                    'size': '20K videos',
                    'url': 'http://activity-net.org',
                    'use': 'Activity recognition (for CCTV)'
                }
            }
        }
    
    @staticmethod
    def synthetic_data_apis():
        """
        Free APIs for synthetic data generation
        """
        return {
            'faker': {
                'type': 'Python library',
                'install': 'pip install faker',
                'use': 'Generate fake invoices, names, addresses',
                'example': 'from faker import Faker; fake = Faker()'
            },
            'imgaug': {
                'type': 'Image augmentation',
                'install': 'pip install imgaug',
                'use': 'Create variations of images',
                'free': 'Unlimited'
            },
            'albumentations': {
                'type': 'Image augmentation',
                'install': 'pip install albumentations',
                'use': 'Fast augmentation for OCR',
                'free': 'Unlimited'
            }
        }


class FreeModelHosting:
    """
    Free model hosting and inference
    """
    
    @staticmethod
    def hosting_options():
        """
        Free hosting platforms
        """
        return {
            'huggingface_spaces': {
                'free_tier': 'CPU/GPU (limited)',
                'cost': 'Free (public), $9/month (private GPU)',
                'deployment': 'Gradio/Streamlit app',
                'url': 'https://huggingface.co/spaces',
                'best_for': 'Demos and inference'
            },
            'replicate': {
                'free_tier': '$10 credits on signup',
                'cost': 'Pay per inference',
                'deployment': 'Docker container',
                'url': 'https://replicate.com',
                'best_for': 'Production inference API'
            },
            'modal': {
                'free_tier': '$30/month credits',
                'cost': 'Pay as you go',
                'deployment': 'Python functions',
                'url': 'https://modal.com',
                'best_for': 'Serverless GPU inference'
            },
            'railway': {
                'free_tier': '$5/month credits',
                'cost': 'Pay as you go',
                'deployment': 'Docker/GitHub',
                'url': 'https://railway.app',
                'best_for': 'Web apps with API'
            }
        }


def optimize_training_with_free_apis():
    """
    Strategy to optimize training using free resources
    """
    
    strategy = {
        'data_labeling': {
            'method': 'Use Gemini API for caption generation',
            'cost': 'Free (1500/day)',
            'code': '''
# Generate captions with Gemini
api = FreeAPIIntegration()
for image in unlabeled_images:
    caption = api.gemini_vision_api(image, "Describe in detail")
    save_caption(image, caption)
'''
        },
        
        'validation': {
            'method': 'Use Hugging Face Inference API',
            'cost': 'Free (1000/month)',
            'code': '''
# Validate with BLIP-2
api = FreeAPIIntegration()
for val_image in validation_set:
    pred_caption = api.huggingface_inference(val_image, "Salesforce/blip2-opt-2.7b")
    compare_with_ground_truth(pred_caption)
'''
        },
        
        'training': {
            'method': 'Rotate between free GPU platforms',
            'schedule': {
                'week1': 'Google Colab (50 hours)',
                'week2': 'Kaggle (30 hours)',
                'week3': 'Lightning AI (22 hours)',
                'week4': 'Mix of all'
            },
            'total_free': '100+ hours/month'
        },
        
        'inference_deployment': {
            'method': 'Deploy on Hugging Face Spaces',
            'cost': 'Free (CPU) or $9/month (GPU)',
            'code': '''
# Deploy Gradio app to HF Spaces
# Create app.py with your model
# Push to HF: git push https://huggingface.co/spaces/YOUR_NAME/assistbuddy
'''
        },
        
        'synthetic_data': {
            'method': 'Use Faker + image augmentation',
            'cost': 'Free unlimited',
            'code': '''
from faker import Faker
import albumentations as A

fake = Faker()
# Generate 1000 fake invoices
for i in range(1000):
    invoice_data = {
        'company': fake.company(),
        'amount': fake.random_int(1000, 100000),
        'date': fake.date()
    }
    create_invoice_image(invoice_data)
'''
        }
    }
    
    return strategy


# Usage example
if __name__ == "__main__":
    print("="*70)
    print("FREE RESOURCES FOR MULTIMODAL TRAINING")
    print("="*70 + "\n")
    
    # Show free APIs
    print("1. FREE APIs:")
    print("-" * 70)
    api = FreeAPIIntegration()
    print("✓ Hugging Face Inference: 1000 requests/month")
    print("✓ Google Gemini: 1500 requests/day (FREE)")
    print("✓ Replicate: $10 signup credits")
    print("✓ OpenRouter: Free models available")
    
    # Show free GPU platforms
    print("\n2. FREE GPU PLATFORMS:")
    print("-" * 70)
    gpus = FreeGPUResources.get_available_platforms()
    for name, info in gpus.items():
        print(f"\n{name.upper().replace('_', ' ')}:")
        print(f"  GPU: {info['gpu']}")
        print(f"  Free hours: {info['free_hours']}")
        print(f"  URL: {info['url']}")
    
    strategy = FreeGPUResources.best_strategy()
    print(f"\n📊 OPTIMAL STRATEGY: {strategy['total_free_hours']} free GPU time/week!")
    print(f"💰 Cost: {strategy['cost']}")
    
    # Show datasets
    print("\n3. FREE DATASETS:")
    print("-" * 70)
    datasets = FreeDatasetResources.get_free_datasets()
    print(f"✓ COCO: {datasets['image_caption']['coco_captions']['size']}")
    print(f"✓ DocVQA: {datasets['document_understanding']['docvqa']['size']}")
    print(f"✓ LAION-400M: {datasets['image_caption']['laion_400m']['size']}")
    
    # Show optimization strategy
    print("\n4. OPTIMIZATION STRATEGY:")
    print("-" * 70)
    opt_strategy = optimize_training_with_free_apis()
    print("\n Data Labeling: Use Gemini API (1500 free/day)")
    print(" Training: Rotate GPUs (100+ hours/month free)")
    print(" Deployment: Hugging Face Spaces (free tier)")
    print(" Synthetic Data: Faker + augmentation (unlimited)")
    
    print("\n" + "="*70)
    print("TOTAL MONTHLY COST: $0 🎉")
    print("="*70)
