"""
Vision-Language Model (VLM) Integration
Alternative to from-scratch training using pre-trained VLMs
"""

import torch
import torch.nn as nn
from transformers import (
    # LLaVA
    LlavaForConditionalGeneration,
    AutoProcessor as LlavaProcessor,
    
    # BLIP-2
    Blip2ForConditionalGeneration,
    Blip2Processor,
    
    # InstructBLIP
    InstructBlipForConditionalGeneration,
    InstructBlipProcessor,
    
    # Qwen-VL
    AutoModelForCausalLM,
    AutoTokenizer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from typing import Optional, Dict


class VLMBasedAssistBuddy(nn.Module):
    """
    AssistBuddy using pre-trained VLMs (faster, better performance)
    
    Supported VLMs:
    - LLaVA: llava-hf/llava-1.5-7b-hf
    - BLIP-2: Salesforce/blip2-opt-2.7b
    - InstructBLIP: Salesforce/instructblip-vicuna-7b
    - Qwen-VL: Qwen/Qwen-VL-Chat
    
    Advantages over from-scratch:
    - Faster training (fine-tuning instead of training from scratch)
    - Better initial performance (already trained on vision-language tasks)
    - Smaller dataset needed (can work with 100-500 samples)
    - Lower GPU requirements (LoRA fine-tuning)
    """
    
    def __init__(
        self,
        vlm_type: str = "llava",  # "llava", "blip2", "instructblip", "qwen-vl"
        model_size: str = "7b",  # "7b", "13b" (not all models have all sizes)
        use_lora: bool = True,  # Use LoRA for memory-efficient fine-tuning
        lora_r: int = 8,
        lora_alpha: int = 16,
        load_in_8bit: bool = False  # Use 8-bit quantization (even more memory efficient)
    ):
        super().__init__()
        
        self.vlm_type = vlm_type
        self.use_lora = use_lora
        
        print(f"Loading {vlm_type.upper()} model...")
        
        # Load appropriate VLM
        if vlm_type == "llava":
            model_name = f"llava-hf/llava-1.5-{model_size}-hf"
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                load_in_8bit=load_in_8bit,
                device_map="auto"
            )
            self.processor = LlavaProcessor.from_pretrained(model_name)
            
        elif vlm_type == "blip2":
            model_name = f"Salesforce/blip2-opt-{model_size}"
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                load_in_8bit=load_in_8bit,
                device_map="auto"
            )
            self.processor = Blip2Processor.from_pretrained(model_name)
            
        elif vlm_type == "instructblip":
            model_name = f"Salesforce/instructblip-vicuna-{model_size}"
            self.model = InstructBlipForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                load_in_8bit=load_in_8bit,
                device_map="auto"
            )
            self.processor = InstructBlipProcessor.from_pretrained(model_name)
            
        elif vlm_type == "qwen-vl":
            model_name = "Qwen/Qwen-VL-Chat"
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                load_in_8bit=load_in_8bit,
                device_map="auto"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            self.processor = None  # Qwen uses tokenizer directly
        
        else:
            raise ValueError(f"Unknown VLM type: {vlm_type}")
        
        print(f"✓ Loaded {model_name}")
        
        # Apply LoRA for parameter-efficient fine-tuning
        if use_lora:
            print("Applying LoRA for efficient fine-tuning...")
            
            # Prepare model for training
            if load_in_8bit:
                self.model = prepare_model_for_kbit_training(self.model)
            
            # LoRA configuration
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=self._get_lora_target_modules(vlm_type),
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        
        # Style prompts for ADMIN/FRIEND
        self.style_prompts = {
            'admin': "Generate a professional admin summary with TL;DR, Key Details, and Actions & Risks. Include provenance and confidence scores.",
            'friend': "Generate a casual, friendly summary like chatting with a friend. Use short sentences, Hinglish if natural (yaar, boss, thoda). Still include sources."
        }
    
    def _get_lora_target_modules(self, vlm_type: str):
        """Get module names to apply LoRA to"""
        if vlm_type == "llava":
            return ["q_proj", "v_proj", "k_proj", "o_proj"]
        elif vlm_type in ["blip2", "instructblip"]:
            return ["q_proj", "v_proj"]
        elif vlm_type == "qwen-vl":
            return ["c_attn", "c_proj", "w1", "w2"]
        return ["q_proj", "v_proj"]
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_text: str,
        labels: Optional[torch.Tensor] = None,
        style: str = 'admin'
    ):
        """
        Forward pass through VLM
        
        Args:
            pixel_values: Image tensor
            input_text: Optional prompt text
            labels: Target labels for training
            style: 'admin' or 'friend'
        """
        # Construct prompt with style instruction
        prompt = f"{self.style_prompts[style]}\n\nFile: {input_text}"
        
        if self.vlm_type == "qwen-vl":
            # Qwen-VL specific preprocessing
            query = self.tokenizer.from_list_format([
                {'image': pixel_values},
                {'text': prompt}
            ])
            inputs = self.tokenizer(query, return_tensors='pt')
        else:
            # Standard VLM preprocessing
            inputs = self.processor(
                text=prompt,
                images=pixel_values,
                return_tensors="pt",
                padding=True
            )
        
        # Move to device
        inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        
        # Forward
        if labels is not None:
            inputs['labels'] = labels
        
        outputs = self.model(**inputs)
        return outputs
    
    def generate_summary(
        self,
        image: torch.Tensor,
        style: str = 'admin',
        max_new_tokens: int = 512,
        temperature: float = 0.7
    ):
        """
        Generate summary for image
        
        Args:
            image: Image tensor
            style: 'admin' or 'friend'
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated summary text
        """
        self.model.eval()
        
        prompt = self.style_prompts[style]
        
        with torch.no_grad():
            if self.vlm_type == "qwen-vl":
                query = self.tokenizer.from_list_format([
                    {'image': image},
                    {'text': prompt}
                ])
                inputs = self.tokenizer(query, return_tensors='pt')
            else:
                inputs = self.processor(
                    text=prompt,
                    images=image,
                    return_tensors="pt"
                )
            
            inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            
            # Generate
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True
            )
            
            # Decode
            if self.vlm_type == "qwen-vl":
                summary = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            else:
                summary = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )[0]
        
        return summary
    
    def get_trainable_parameters(self):
        """Get count of trainable parameters"""
        if self.use_lora:
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            return {
                'trainable': f"{trainable / 1e6:.1f}M",
                'total': f"{total / 1e6:.1f}M",
                'trainable_pct': f"{100 * trainable / total:.2f}%"
            }
        return {}


def compare_vlm_models():
    """
    Compare different VLM options
    
    Returns comparison data
    """
    
    comparison = {
        'LLaVA-1.5-7B': {
            'parameters': '7B',
            'training_approach': 'Fine-tune with LoRA',
            'gpu_memory_required': '~12GB (with 8-bit)',
            'training_time': '1-2 hours (100 samples)',
            'best_for': 'General multimodal tasks',
            'accuracy_estimate': 'Very High',
            'dataset_size_needed': '100-500 samples',
            'huggingface_model': 'llava-hf/llava-1.5-7b-hf'
        },
        'BLIP-2-OPT-2.7B': {
            'parameters': '2.7B',
            'training_approach': 'Fine-tune with LoRA',
            'gpu_memory_required': '~8GB (with 8-bit)',
            'training_time': '30-60 min (100 samples)',
            'best_for': 'Image captioning, Q&A',
            'accuracy_estimate': 'High',
            'dataset_size_needed': '100-300 samples',
            'huggingface_model': 'Salesforce/blip2-opt-2.7b'
        },
        'InstructBLIP-7B': {
            'parameters': '7B',
            'training_approach': 'Fine-tune with LoRA',
            'gpu_memory_required': '~12GB (with 8-bit)',
            'training_time': '1-2 hours (100 samples)',
            'best_for': 'Instruction following',
            'accuracy_estimate': 'Very High',
            'dataset_size_needed': '100-500 samples',
            'huggingface_model': 'Salesforce/instructblip-vicuna-7b'
        },
        'Qwen-VL-Chat': {
            'parameters': '7B',
            'training_approach': 'Fine-tune with LoRA',
            'gpu_memory_required': '~12GB (with 8-bit)',
            'training_time': '1-2 hours (100 samples)',
            'best_for': 'Multilingual, Chinese support',
            'accuracy_estimate': 'Very High',
            'dataset_size_needed': '100-500 samples',
            'huggingface_model': 'Qwen/Qwen-VL-Chat'
        },
        'From-Scratch-1.3B': {
            'parameters': '1.3B',
            'training_approach': 'Train from scratch',
            'gpu_memory_required': '~15GB',
            'training_time': '4-8 hours (1000 samples)',
            'best_for': 'Full customization, domain-specific',
            'accuracy_estimate': 'Medium (needs large dataset)',
            'dataset_size_needed': '1000-5000 samples',
            'huggingface_model': 'Custom (CLIP+BERT+GPT-2)'
        }
    }
    
    return comparison


# Usage example
if __name__ == "__main__":
    import json
    
    # Show comparison
    models = compare_vlm_models()
    print("VLM Model Comparison:\n")
    print(json.dumps(models, indent=2))
    
    # Test VLM
    print("\n" + "="*60)
    print("Testing VLM Integration")
    print("="*60 + "\n")
    
    print("Loading LLaVA with LoRA...")
    vlm = VLMBasedAssistBuddy(
        vlm_type="llava",
        model_size="7b",
        use_lora=True,
        load_in_8bit=True  # Use 8-bit for memory efficiency
    )
    
    params = vlm.get_trainable_parameters()
    print(f"\n✓ Model ready")
    print(f"  Trainable: {params['trainable']} ({params['trainable_pct']})")
    print(f"  Total: {params['total']}")
