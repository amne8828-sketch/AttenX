import torch
import torch.nn as nn
# from transformers import WhisperModel, WhisperProcessor


class AudioEncoder(nn.Module):
    
    def __init__(
        self,
        model_name: str = "openai/whisper-small",
        output_dim: int = 768,
        num_output_tokens: int = 16,
        quantize: bool = False
    ):
        super().__init__()
        
        # Lazy load transformers
        from transformers import WhisperModel, WhisperProcessor
        
        # Load pretrained Whisper
        self.whisper = WhisperModel.from_pretrained(model_name)
        self.processor = WhisperProcessor.from_pretrained(model_name)
        
        # Quantization
        if quantize:
            self.whisper = torch.quantization.quantize_dynamic(
                self.whisper, {nn.Linear}, dtype=torch.qint8
            )
        
        whisper_hidden_dim = self.whisper.config.d_model  # 768 for small
        
        # Projection layer
        self.projection = nn.Linear(whisper_hidden_dim, output_dim)
        
        # Learnable query tokens for pooling
        self.query_tokens = nn.Parameter(
            torch.randn(1, num_output_tokens, output_dim)
        )
        
        # Cross-attention to compress audio features
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=8,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, audio_features: torch.Tensor):
        
        batch_size = audio_features.shape[0]
        
        # Get encoder outputs from Whisper
        encoder_outputs = self.whisper.encoder(
            audio_features,
            return_dict=True
        )
        audio_hidden = encoder_outputs.last_hidden_state  # [B, seq_len, whisper_hidden_dim]
        
        # Project to target dimension
        audio_hidden = self.projection(audio_hidden)  # [B, seq_len, output_dim]
        
        # Use cross-attention with learnable queries to compress
        queries = self.query_tokens.expand(batch_size, -1, -1)  # [B, num_output_tokens, output_dim]
        
        compressed_features, _ = self.cross_attention(
            query=queries,
            key=audio_hidden,
            value=audio_hidden
        )  # [B, num_output_tokens, output_dim]
        
        # Layer norm
        output = self.layer_norm(compressed_features)
        
        return output
    
    def process_audio(self, audio_array, sampling_rate: int = 16000):
        
        inputs = self.processor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt"
        )
        return inputs.input_features
    
    def save_pretrained(self, save_directory: str):
        """Save audio encoder"""
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        self.whisper.save_pretrained(save_directory)
        torch.save({
            'projection': self.projection.state_dict(),
            'query_tokens': self.query_tokens,
            'cross_attention': self.cross_attention.state_dict(),
            'layer_norm': self.layer_norm.state_dict()
        }, os.path.join(save_directory, 'audio_encoder_extra.pt'))
        
    @classmethod
    def from_pretrained(cls, load_directory: str, **kwargs):
        """Load audio encoder"""
        import os
        
        encoder = cls(**kwargs)
        encoder.whisper = WhisperModel.from_pretrained(load_directory)
        
        extra_weights = torch.load(os.path.join(load_directory, 'audio_encoder_extra.pt'))
        encoder.projection.load_state_dict(extra_weights['projection'])
        encoder.query_tokens = extra_weights['query_tokens']
        encoder.cross_attention.load_state_dict(extra_weights['cross_attention'])
        encoder.layer_norm.load_state_dict(extra_weights['layer_norm'])
        
        return encoder
