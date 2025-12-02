
import torch
import torch.nn as nn
# from transformers import CLIPVisionModel, CLIPImageProcessor


class VisionEncoder(nn.Module):
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch16",
        hidden_dim: int = 768,
        output_dim: int = 768,
        num_output_tokens: int = 16,
        quantize: bool = False
    ):
        super().__init__()
        
        # Lazy load transformers
        from transformers import CLIPVisionModel, CLIPImageProcessor
        
        # Load pretrained CLIP vision model
        self.vision_model = CLIPVisionModel.from_pretrained(model_name)
        self.processor = CLIPImageProcessor.from_pretrained(model_name)
        
        # Quantization
        if quantize:
            self.vision_model = torch.quantization.quantize_dynamic(
                self.vision_model, {nn.Linear}, dtype=torch.qint8
            )
        
        # Get the vision model's hidden dimension
        vision_hidden_dim = self.vision_model.config.hidden_size  # 768 for ViT-B
        
        # Projection layer to match decoder dimension
        self.projection = nn.Linear(vision_hidden_dim, output_dim)
        
        # Learnable query tokens for pooling
        self.query_tokens = nn.Parameter(
            torch.randn(1, num_output_tokens, output_dim)
        )
        
        # Cross-attention to compress vision features
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=8,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, pixel_values: torch.Tensor):
        """
        Encode images to embeddings
        
        Args:
            pixel_values: [batch_size, 3, 224, 224]
            
        Returns:
            vision_embeds: [batch_size, num_output_tokens, output_dim]
        """
        batch_size = pixel_values.shape[0]
        
        # Get vision features from CLIP
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        vision_features = vision_outputs.last_hidden_state  # [B, 197, 768] for ViT-B
        
        # Project to target dimension
        vision_features = self.projection(vision_features)  # [B, 197, output_dim]
        
        # Use cross-attention with learnable queries to compress
        queries = self.query_tokens.expand(batch_size, -1, -1)  # [B, num_output_tokens, output_dim]
        
        compressed_features, _ = self.cross_attention(
            query=queries,
            key=vision_features,
            value=vision_features
        )  # [B, num_output_tokens, output_dim]
        
        # Layer norm
        output = self.layer_norm(compressed_features)
        
        return output
    
    def save_pretrained(self, save_directory: str):
        """Save vision encoder"""
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        self.vision_model.save_pretrained(save_directory)
        torch.save({
            'projection': self.projection.state_dict(),
            'query_tokens': self.query_tokens,
            'cross_attention': self.cross_attention.state_dict(),
            'layer_norm': self.layer_norm.state_dict()
        }, os.path.join(save_directory, 'vision_encoder_extra.pt'))
        
    @classmethod
    def from_pretrained(cls, load_directory: str, **kwargs):
        """Load vision encoder"""
        import os
        
        encoder = cls(**kwargs)
        encoder.vision_model = CLIPVisionModel.from_pretrained(load_directory)
        
        extra_weights = torch.load(os.path.join(load_directory, 'vision_encoder_extra.pt'))
        encoder.projection.load_state_dict(extra_weights['projection'])
        encoder.query_tokens = extra_weights['query_tokens']
        encoder.cross_attention.load_state_dict(extra_weights['cross_attention'])
        encoder.layer_norm.load_state_dict(extra_weights['layer_norm'])
        
        return encoder
