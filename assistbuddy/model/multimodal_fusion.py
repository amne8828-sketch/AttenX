import torch
import torch.nn as nn
from typing import Optional


class MultimodalFusion(nn.Module):
    
    
    def __init__(
        self,
        hidden_dim: int = 768,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Modality type embeddings (like segment embeddings in BERT)
        self.modality_embeddings = nn.Embedding(3, hidden_dim)  # vision=0, text=1, audio=2
        
        # Transformer encoder for cross-modal fusion
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        modality_ids: Optional[torch.Tensor] = None
    ):
        
        # Add modality type embeddings if provided
        if modality_ids is not None:
            modality_embeds = self.modality_embeddings(modality_ids)
            embeddings = embeddings + modality_embeds
        
        # Create attention mask in the format Transformer expects
        # Transformer needs: [batch_size, seq_len] with True for positions to mask
        if attention_mask is not None:
            # Convert from [B, L] with 1=attend to [B, L] with True=ignore
            mask = (attention_mask == 0)
        else:
            mask = None
        
        # Apply transformer fusion
        fused = self.transformer(
            embeddings,
            src_key_padding_mask=mask
        )
        
        # Layer norm
        output = self.layer_norm(fused)
        
        return output


class SimpleConcatFusion(nn.Module):
    
    
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        modality_ids: Optional[torch.Tensor] = None
    ):
        """Just normalize, no cross-attention"""
        return self.layer_norm(embeddings)
