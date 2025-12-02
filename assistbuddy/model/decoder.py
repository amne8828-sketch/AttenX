import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config


class StyleControlledDecoder(nn.Module):
    
    
    def __init__(
        self,
        model_name: str = "gpt2",  # gpt2, gpt2-medium, gpt2-large
        hidden_dim: int = 768,
        add_special_tokens: bool = True
    ):
        super().__init__()
        
        # Load pretrained GPT-2
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        
        # Add special tokens for style control
        if add_special_tokens:
            special_tokens = {
                'pad_token': '[PAD]',
                'additional_special_tokens': ['[ADMIN]', '[FRIEND]', '[REDACTED]']
            }
            self.tokenizer.add_special_tokens(special_tokens)
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Input projection layer to map multimodal embeddings to GPT-2 space
        gpt2_hidden_dim = self.model.config.n_embd  # 768 for gpt2
        
        if hidden_dim != gpt2_hidden_dim:
            self.input_projection = nn.Linear(hidden_dim, gpt2_hidden_dim)
        else:
            self.input_projection = nn.Identity()
            
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
        return_dict: bool = True
    ):
        
        # Project multimodal embeddings to GPT-2 space
        inputs_embeds = self.input_projection(inputs_embeds)
        
        # Generate
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=return_dict
        )
        
        return outputs
    
    def generate(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_beams: int = 1,
        **kwargs
    ):
        
        # Project to GPT-2 space
        inputs_embeds = self.input_projection(inputs_embeds)
        
        # Generate
        outputs = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs
        )
        
        return outputs
    
    def save_pretrained(self, save_directory: str):
        """Save decoder"""
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        
        if not isinstance(self.input_projection, nn.Identity):
            torch.save(
                self.input_projection.state_dict(),
                os.path.join(save_directory, 'input_projection.pt')
            )
    
    @classmethod
    def from_pretrained(cls, load_directory: str, **kwargs):
        """Load decoder"""
        import os
        
        decoder = cls(**kwargs)
        decoder.model = GPT2LMHeadModel.from_pretrained(load_directory)
        decoder.tokenizer = GPT2Tokenizer.from_pretrained(load_directory)
        
        projection_path = os.path.join(load_directory, 'input_projection.pt')
        if os.path.exists(projection_path):
            decoder.input_projection.load_state_dict(torch.load(projection_path))
            
        return decoder
