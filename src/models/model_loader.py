#!/usr/bin/env python3
"""
Model loading utilities for context manipulation experiments.
Supports HuggingFace models and optional API-based models.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, Optional, Dict, Any


class ModelWrapper:
    """
    Wrapper class for language models with unified interface.
    Supports both local HF models and API-based models.
    """
    
    def __init__(
        self,
        model: Optional[AutoModelForCausalLM] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        model_name: str = "",
        api_type: Optional[str] = None
    ):
        """
        Initialize model wrapper.
        
        Args:
            model: HuggingFace model instance
            tokenizer: HuggingFace tokenizer instance
            model_name: Name/identifier of the model
            api_type: Type of API if using hosted model ('openai', 'anthropic', etc.)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.api_type = api_type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if model is not None:
            self.model.to(self.device)
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs
    ) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling parameter
            do_sample: Whether to use sampling
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text (full output including prompt)
        """
        if self.api_type:
            return self._generate_api(prompt, max_new_tokens, temperature, **kwargs)
        else:
            return self._generate_local(prompt, max_new_tokens, temperature, top_p, do_sample, **kwargs)
    
    def _generate_local(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
        **kwargs
    ) -> str:
        """Generate using local HuggingFace model."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    
    def _generate_api(self, prompt: str, max_tokens: int, temperature: float, **kwargs) -> str:
        """Generate using API-based model (OpenAI, Anthropic, etc.)."""
        if self.api_type == "openai":
            return self._generate_openai(prompt, max_tokens, temperature, **kwargs)
        elif self.api_type == "anthropic":
            return self._generate_anthropic(prompt, max_tokens, temperature, **kwargs)
        else:
            raise ValueError(f"Unsupported API type: {self.api_type}")
    
    def _generate_openai(self, prompt: str, max_tokens: int, temperature: float, **kwargs) -> str:
        """Generate using OpenAI API."""
        try:
            import openai
        except ImportError:
            raise ImportError("openai package not installed. Install with: pip install openai")
        
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        return response.choices[0].message.content
    
    def _generate_anthropic(self, prompt: str, max_tokens: int, temperature: float, **kwargs) -> str:
        """Generate using Anthropic API."""
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package not installed. Install with: pip install anthropic")
        
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.content[0].text
    
    def get_logits(self, prompt: str) -> torch.Tensor:
        """
        Get logits for a given prompt (local models only).
        
        Args:
            prompt: Input text
            
        Returns:
            Logits tensor
        """
        if self.api_type:
            raise NotImplementedError("Logits not available for API-based models")
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.logits
    
    def get_perplexity(self, text: str) -> float:
        """
        Calculate perplexity of text (local models only).
        
        Args:
            text: Input text
            
        Returns:
            Perplexity score
        """
        if self.api_type:
            raise NotImplementedError("Perplexity not available for API-based models")
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs.input_ids)
            loss = outputs.loss
        
        perplexity = torch.exp(loss).item()
        return perplexity


def load_model(
    model_name: str = "google/gemma-2-2b",
    device: Optional[str] = None,
    load_in_8bit: bool = False,
    torch_dtype: Optional[torch.dtype] = None,
    api_key: Optional[str] = None
) -> Tuple[ModelWrapper, Optional[AutoTokenizer]]:
    """
    Load a language model for experimentation.
    
    Args:
        model_name: HuggingFace model name or API model identifier
        device: Device to load model on ('cuda', 'cpu', or None for auto)
        load_in_8bit: Whether to load in 8-bit precision (requires bitsandbytes)
        torch_dtype: PyTorch dtype for model weights
        api_key: API key for hosted models
        
    Returns:
        Tuple of (ModelWrapper, tokenizer)
    """
    # Determine if this is an API-based model
    api_models = {
        "gpt-4": "openai",
        "gpt-3.5-turbo": "openai",
        "claude-3": "anthropic",
        "claude-2": "anthropic"
    }
    
    api_type = None
    for model_prefix, api_name in api_models.items():
        if model_prefix in model_name:
            api_type = api_name
            break
    
    if api_type:
        # API-based model
        if api_key:
            if api_type == "openai":
                import openai
                openai.api_key = api_key
            elif api_type == "anthropic":
                import anthropic
                # Anthropic client handles key via environment or constructor
        
        wrapper = ModelWrapper(
            model=None,
            tokenizer=None,
            model_name=model_name,
            api_type=api_type
        )
        return wrapper, None
    
    # Local HuggingFace model
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if torch_dtype is None:
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
    
    print(f"Loading model: {model_name}")
    print(f"Device: {device}, dtype: {torch_dtype}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": "auto" if device == "cuda" else None,
    }
    
    if load_in_8bit:
        model_kwargs["load_in_8bit"] = True
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )
    
    if not load_in_8bit and device != "cuda":
        model = model.to(device)
    
    model.eval()
    
    wrapper = ModelWrapper(
        model=model,
        tokenizer=tokenizer,
        model_name=model_name,
        api_type=None
    )
    
    return wrapper, tokenizer


if __name__ == "__main__":
    # Test loading
    wrapper, tokenizer = load_model("gpt2")
    output = wrapper.generate("Hello, world!", max_new_tokens=20)
    print(f"Test output: {output}")

