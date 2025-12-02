"""
Text Encoder for extracted text from PDFs, Word, Excel
Uses BERT as backbone
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class TextEncoder(nn.Module):
    """
    Text encoder using BERT
    Encodes extracted text to embeddings
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        output_dim: int = 768,
        max_length: int = 512
    ):
        super().__init__()
        
        # Load pretrained BERT
        self.bert = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        bert_hidden_dim = self.bert.config.hidden_size  # 768
        
        # Projection layer if needed
        if bert_hidden_dim != output_dim:
            self.projection = nn.Linear(bert_hidden_dim, output_dim)
        else:
            self.projection = nn.Identity()
            
        self.max_length = max_length
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        Encode text to embeddings
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            BaseModelOutputWithPoolingAndCrossAttentions with:
                - last_hidden_state: [batch_size, seq_len, output_dim]
        """
        # Get BERT embeddings
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Project to target dimension
        hidden_states = self.projection(bert_outputs.last_hidden_state)
        
        # Return in same format as BERT output
        bert_outputs.last_hidden_state = hidden_states
        
        return bert_outputs
    
    def encode_text(self, texts: list[str], device: str = 'cuda'):
        """
        Convenience method to tokenize and encode text
        
        Args:
            texts: List of text strings
            device: Device to put tensors on
            
        Returns:
            Dictionary with input_ids, attention_mask, embeddings
        """
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        # Encode
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'embeddings': outputs.last_hidden_state
        }
    
    def save_pretrained(self, save_directory: str):
        """Save text encoder"""
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        self.bert.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        
        if not isinstance(self.projection, nn.Identity):
            torch.save(
                self.projection.state_dict(),
                os.path.join(save_directory, 'projection.pt')
            )
    
    @classmethod
    def from_pretrained(cls, load_directory: str, **kwargs):
        """Load text encoder"""
        import os
        
        encoder = cls(**kwargs)
        encoder.bert = BertModel.from_pretrained(load_directory)
        encoder.tokenizer = BertTokenizer.from_pretrained(load_directory)
        
        projection_path = os.path.join(load_directory, 'projection.pt')
        if os.path.exists(projection_path):
            encoder.projection.load_state_dict(torch.load(projection_path))
            
        return encoder
