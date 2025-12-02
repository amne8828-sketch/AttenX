"""
AssistBuddy Main Model
Multimodal encoder-decoder architecture for file summarization
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer


class AssistBuddyModel(nn.Module):
    """
    Main AssistBuddy model combining:
    - Vision encoder (CLIP/ViT)
    - Text encoder (BERT)  
    - Audio encoder (Whisper)
    - Multimodal fusion layer
    - Language decoder (GPT-style)
    """
    
    def __init__(
        self,
        vision_encoder,
        text_encoder,
        audio_encoder,
        fusion_layer,
        decoder,
        tokenizer,
        max_length: int = 512,
        style_token_ids: Optional[Dict[str, int]] = None
    ):
        super().__init__()
        
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.audio_encoder = audio_encoder
        self.fusion_layer = fusion_layer
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Special tokens for style control
        self.style_token_ids = style_token_ids or {
            'admin': tokenizer.convert_tokens_to_ids('[ADMIN]'),
            'friend': tokenizer.convert_tokens_to_ids('[FRIEND]')
        }
        
    def forward(
        self,
        image_inputs: Optional[torch.Tensor] = None,
        text_inputs: Optional[Dict[str, torch.Tensor]] = None,
        audio_inputs: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        style: str = 'admin',
        return_dict: bool = True
    ):
        """
        Forward pass through multimodal model
        
        Args:
            image_inputs: Batch of images [B, C, H, W]
            text_inputs: Dict with input_ids, attention_mask
            audio_inputs: Batch of audio spectrograms [B, T, F]
            labels: Target token IDs for training
            style: 'admin' or 'friend'
            
        Returns:
            ModelOutput with loss, logits, hidden_states
        """
        batch_size = self._get_batch_size(image_inputs, text_inputs, audio_inputs)
        device = self._get_device(image_inputs, text_inputs, audio_inputs)
        
        # Encode each modality
        embeddings_list = []
        attention_masks = []
        
        # Vision encoding
        if image_inputs is not None:
            vision_embeds = self.vision_encoder(image_inputs)  # [B, seq_len, hidden_dim]
            embeddings_list.append(vision_embeds)
            attention_masks.append(torch.ones(vision_embeds.shape[:2], device=device))
            
        # Text encoding
        if text_inputs is not None:
            text_embeds = self.text_encoder(
                input_ids=text_inputs['input_ids'],
                attention_mask=text_inputs['attention_mask']
            ).last_hidden_state  # [B, seq_len, hidden_dim]
            embeddings_list.append(text_embeds)
            attention_masks.append(text_inputs['attention_mask'])
            
        # Audio encoding
        if audio_inputs is not None:
            audio_embeds = self.audio_encoder(audio_inputs)  # [B, seq_len, hidden_dim]
            embeddings_list.append(audio_embeds)
            attention_masks.append(torch.ones(audio_embeds.shape[:2], device=device))
        
        # Fusion: concatenate all modalities
        if len(embeddings_list) > 0:
            fused_embeds = torch.cat(embeddings_list, dim=1)  # [B, total_seq_len, hidden_dim]
            fused_attention = torch.cat(attention_masks, dim=1)  # [B, total_seq_len]
            
            # Apply fusion layer (cross-modal attention)
            fused_embeds = self.fusion_layer(
                fused_embeds,
                attention_mask=fused_attention
            )
        else:
            raise ValueError("At least one modality input must be provided")
        
        # Add style token
        style_token_id = self.style_token_ids[style]
        style_embeds = self.decoder.model.embed_tokens(
            torch.tensor([[style_token_id]], device=device).expand(batch_size, 1)
        )
        
        # Prepend style token to fused embeddings
        decoder_inputs_embeds = torch.cat([style_embeds, fused_embeds], dim=1)
        decoder_attention_mask = torch.cat([
            torch.ones(batch_size, 1, device=device),
            fused_attention
        ], dim=1)
        
        # Decode
        outputs = self.decoder(
            inputs_embeds=decoder_inputs_embeds,
            attention_mask=decoder_attention_mask,
            labels=labels,
            return_dict=return_dict
        )
        
        return outputs
    
    def generate(
        self,
        image_inputs: Optional[torch.Tensor] = None,
        text_inputs: Optional[Dict[str, torch.Tensor]] = None,
        audio_inputs: Optional[torch.Tensor] = None,
        style: str = 'admin',
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_beams: int = 1
    ):
        """
        Generate summary text
        
        Returns:
            Generated text string
        """
        self.eval()
        
        with torch.no_grad():
            batch_size = self._get_batch_size(image_inputs, text_inputs, audio_inputs)
            device = self._get_device(image_inputs, text_inputs, audio_inputs)
            
            # Encode modalities
            embeddings_list = []
            attention_masks = []
            
            if image_inputs is not None:
                vision_embeds = self.vision_encoder(image_inputs)
                embeddings_list.append(vision_embeds)
                attention_masks.append(torch.ones(vision_embeds.shape[:2], device=device))
                
            if text_inputs is not None:
                text_embeds = self.text_encoder(
                    input_ids=text_inputs['input_ids'],
                    attention_mask=text_inputs['attention_mask']
                ).last_hidden_state
                embeddings_list.append(text_embeds)
                attention_masks.append(text_inputs['attention_mask'])
                
            if audio_inputs is not None:
                audio_embeds = self.audio_encoder(audio_inputs)
                embeddings_list.append(audio_embeds)
                attention_masks.append(torch.ones(audio_embeds.shape[:2], device=device))
            
            # Fusion
            fused_embeds = torch.cat(embeddings_list, dim=1)
            fused_attention = torch.cat(attention_masks, dim=1)
            fused_embeds = self.fusion_layer(fused_embeds, attention_mask=fused_attention)
            
            # Add style token
            style_token_id = self.style_token_ids[style]
            style_embeds = self.decoder.model.embed_tokens(
                torch.tensor([[style_token_id]], device=device).expand(batch_size, 1)
            )
            
            decoder_inputs_embeds = torch.cat([style_embeds, fused_embeds], dim=1)
            decoder_attention_mask = torch.cat([
                torch.ones(batch_size, 1, device=device),
                fused_attention
            ], dim=1)
            
            # Generate
            output_ids = self.decoder.generate(
                inputs_embeds=decoder_inputs_embeds,
                attention_mask=decoder_attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            # Decode to text
            generated_text = self.tokenizer.batch_decode(
                output_ids,
                skip_special_tokens=True
            )
            
            return generated_text
    
    def _get_batch_size(self, image_inputs, text_inputs, audio_inputs):
        """Extract batch size from any available input"""
        if image_inputs is not None:
            return image_inputs.shape[0]
        elif text_inputs is not None:
            return text_inputs['input_ids'].shape[0]
        elif audio_inputs is not None:
            return audio_inputs.shape[0]
        raise ValueError("No inputs provided")
    
    def _get_device(self, image_inputs, text_inputs, audio_inputs):
        """Extract device from any available input"""
        if image_inputs is not None:
            return image_inputs.device
        elif text_inputs is not None:
            return text_inputs['input_ids'].device
        elif audio_inputs is not None:
            return audio_inputs.device
        raise ValueError("No inputs provided")
    
    def save_pretrained(self, save_directory: str):
        """Save model weights and config"""
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # Save each component
        self.vision_encoder.save_pretrained(os.path.join(save_directory, "vision_encoder"))
        self.text_encoder.save_pretrained(os.path.join(save_directory, "text_encoder"))
        self.audio_encoder.save_pretrained(os.path.join(save_directory, "audio_encoder"))
        torch.save(self.fusion_layer.state_dict(), os.path.join(save_directory, "fusion_layer.pt"))
        self.decoder.save_pretrained(os.path.join(save_directory, "decoder"))
        self.tokenizer.save_pretrained(os.path.join(save_directory, "tokenizer"))
        
    @classmethod
    def from_pretrained(cls, load_directory: str):
        """Load model from saved weights"""
        # This would be implemented with proper loading logic
        raise NotImplementedError("Loading will be implemented in training notebook")
