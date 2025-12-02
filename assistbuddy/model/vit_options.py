"""
Vision Transformer (ViT) Options for AssistBuddy
Multiple ViT architectures supported
"""

import torch
import torch.nn as nn
from transformers import (
    # CLIP-based ViT (Current default)
    CLIPVisionModel, 
    CLIPImageProcessor,
    
    # Pure ViT models
    ViTModel,
    ViTImageProcessor,
    
    # DeiT (Data-efficient Image Transformer)
    DeiTModel,
    DeiTImageProcessor
)


class VisionTransformerEncoder(nn.Module):
    """
    Vision Transformer encoder with multiple architecture options
    
    Supported models:
    1. CLIP-ViT-B/16 (default) - Best for multimodal understanding
    2. Google ViT-B/16 - Pure vision transformer
    3. DeiT-B - Data-efficient training
    """
    
    def __init__(
        self,
        model_type: str = "clip",  # "clip", "vit", "deit"
        model_size: str = "base",  # "base", "large"
        output_dim: int = 768,
        num_output_tokens: int = 16
    ):
        super().__init__()
        
        self.model_type = model_type
        self.model_size = model_size
        
        # Load appropriate model
        if model_type == "clip":
            model_name = f"openai/clip-vit-{model_size}-patch16"
            self.vision_model = CLIPVisionModel.from_pretrained(model_name)
            self.processor = CLIPImageProcessor.from_pretrained(model_name)
            hidden_dim = self.vision_model.config.hidden_size
            
        elif model_type == "vit":
            model_name = f"google/vit-{model_size}-patch16-224"
            self.vision_model = ViTModel.from_pretrained(model_name)
            self.processor = ViTImageProcessor.from_pretrained(model_name)
            hidden_dim = self.vision_model.config.hidden_size
            
        elif model_type == "deit":
            model_name = f"facebook/deit-{model_size}-patch16-224"
            self.vision_model = DeiTModel.from_pretrained(model_name)
            self.processor = DeiTImageProcessor.from_pretrained(model_name)
            hidden_dim = self.vision_model.config.hidden_size
        
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        print(f"✓ Loaded {model_name}")
        print(f"  Hidden dim: {hidden_dim}")
        
        # Projection to target dimension
        self.projection = nn.Linear(hidden_dim, output_dim)
        
        # Learnable query tokens for pooling
        self.query_tokens = nn.Parameter(
            torch.randn(1, num_output_tokens, output_dim)
        )
        
        # Cross-attention for compression
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=8,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, pixel_values: torch.Tensor):
        """
        Encode images with Vision Transformer
        
        Args:
            pixel_values: [batch_size, 3, 224, 224]
            
        Returns:
            vision_embeds: [batch_size, num_output_tokens, output_dim]
        """
        batch_size = pixel_values.shape[0]
        
        # Get ViT features
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        vision_features = vision_outputs.last_hidden_state  # [B, seq_len, hidden]
        
        # Project
        vision_features = self.projection(vision_features)
        
        # Compress with cross-attention
        queries = self.query_tokens.expand(batch_size, -1, -1)
        
        compressed_features, _ = self.cross_attention(
            query=queries,
            key=vision_features,
            value=vision_features
        )
        
        # Layer norm
        output = self.layer_norm(compressed_features)
        
        return output
    
    def get_model_info(self):
        """Get model architecture information"""
        total_params = sum(p.numel() for p in self.parameters())
        
        return {
            'model_type': self.model_type,
            'model_size': self.model_size,
            'total_parameters': f"{total_params / 1e6:.1f}M",
            'hidden_dim': self.vision_model.config.hidden_size,
            'num_layers': self.vision_model.config.num_hidden_layers,
            'num_attention_heads': self.vision_model.config.num_attention_heads,
            'patch_size': self.vision_model.config.patch_size
        }


# Model comparison
def compare_vit_models():
    """
    Compare different ViT architectures
    
    Returns performance characteristics
    """
    
    comparisons = {
        'clip-vit-base': {
            'parameters': '86M',
            'image_size': 224,
            'patch_size': 16,
            'hidden_dim': 768,
            'layers': 12,
            'heads': 12,
            'best_for': 'Multimodal tasks, zero-shot classification',
            'pretrained_on': 'CLIP (400M image-text pairs)',
            'speed': 'Fast',
            'memory': '~6GB for training'
        },
        'vit-base': {
            'parameters': '86M',
            'image_size': 224,
            'patch_size': 16,
            'hidden_dim': 768,
            'layers': 12,
            'heads': 12,
            'best_for': 'Pure vision tasks, image classification',
            'pretrained_on': 'ImageNet-21k',
            'speed': 'Fast',
            'memory': '~6GB for training'
        },
        'deit-base': {
            'parameters': '86M',
            'image_size': 224,
            'patch_size': 16,
            'hidden_dim': 768,
            'layers': 12,
            'heads': 12,
            'best_for': 'Data-efficient training, distillation',
            'pretrained_on': 'ImageNet-1k (with distillation)',
            'speed': 'Fast',
            'memory': '~6GB for training'
        },
        'clip-vit-large': {
            'parameters': '304M',
            'image_size': 224,
            'patch_size': 14,
            'hidden_dim': 1024,
            'layers': 24,
            'heads': 16,
            'best_for': 'High accuracy, rich representations',
            'pretrained_on': 'CLIP (400M pairs)',
            'speed': 'Medium',
            'memory': '~16GB for training'
        }
    }
    
    return comparisons


# Usage example
if __name__ == "__main__":
    import json
    
    # Compare models
    models = compare_vit_models()
    print("Vision Transformer Model Comparison:\n")
    print(json.dumps(models, indent=2))
    
    # Test each architecture
    print("\n" + "="*60)
    print("Testing ViT Architectures")
    print("="*60 + "\n")
    
    for model_type in ['clip', 'vit', 'deit']:
        print(f"\nTesting {model_type.upper()}...")
        
        encoder = VisionTransformerEncoder(
            model_type=model_type,
            model_size='base',
            output_dim=768
        )
        
        # Test forward pass
        dummy_images = torch.randn(2, 3, 224, 224)
        outputs = encoder(dummy_images)
        
        # Print info
        info = encoder.get_model_info()
        print(f"  Parameters: {info['total_parameters']}")
        print(f"  Output shape: {outputs.shape}")
        print(f"  Layers: {info['num_layers']}")
        print(f"  Attention heads: {info['num_attention_heads']}")
