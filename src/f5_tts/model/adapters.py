"""
Adapter modules for multi-language fine-tuning of F5-TTS.
Supports adding language-specific adapters while preserving base model capabilities.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Union

SUPPORTED_ADAPTER_LANGUAGES = ["ro"]

class LoRAAdapter(nn.Module):
    """
    Low-Rank Adaptation (LoRA) adapter for efficient fine-tuning.
    """
    
    def __init__(self, in_features: int, out_features: int, rank: int = 16, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA parameters
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize B to zero for stable training
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply LoRA adaptation to input."""
        lora_out = F.linear(x, (self.lora_B @ self.lora_A) * self.scaling)
        return self.dropout(lora_out)


class AdapterBlock(nn.Module):
    """
    Adapter block that can be inserted into transformer layers.
    """
    
    def __init__(self, dim: int, bottleneck_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.down_proj = nn.Linear(dim, bottleneck_dim)
        self.up_proj = nn.Linear(bottleneck_dim, dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        # Initialize with small values for stable training
        nn.init.normal_(self.down_proj.weight, std=0.02)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply adapter transformation with residual connection."""
        residual = x
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_proj(x)
        return x + residual


class LanguageAdapter(nn.Module):
    """
    Language-specific adapter that can be applied to attention and feed-forward layers.
    """
    
    def __init__(
        self,
        dim: int,
        language_id: str,
        adapter_type: str = "lora",
        rank: int = 16,
        alpha: float = 16.0,
        bottleneck_dim: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        self.language_id = language_id
        self.adapter_type = adapter_type
        
        if adapter_type == "lora":
            self.adapter = LoRAAdapter(dim, dim, rank, alpha, dropout)
        elif adapter_type == "bottleneck":
            self.adapter = AdapterBlock(dim, bottleneck_dim, dropout)
        else:
            raise ValueError(f"Unknown adapter type: {adapter_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply language-specific adaptation."""
        return self.adapter(x)


class MultiLanguageAdapter(nn.Module):
    """
    Multi-language adapter that routes between different language-specific adapters.
    """
    
    def __init__(
        self,
        dim: int,
        languages: Dict[str, Dict],
        default_language: str = "en"
    ):
        super().__init__()
        self.dim = dim
        self.default_language = default_language
        
        # Create language-specific adapters
        self.language_adapters = nn.ModuleDict()
        for lang_id, config in languages.items():
            self.language_adapters[lang_id] = LanguageAdapter(
                dim=dim,
                language_id=lang_id,
                **config
            )
        
        # Language embedding for conditioning
        self.language_embed = nn.Embedding(len(languages), dim)
        self.language_to_id = {lang: i for i, lang in enumerate(languages.keys())}
    
    def forward(
        self,
        x: torch.Tensor,
        language: Optional[str] = None,
        language_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply language-specific adaptation.
        
        Args:
            x: Input tensor [batch, seq_len, dim]
            language: Language string (e.g., "en", "ro")
            language_ids: Tensor of language IDs [batch]
        """
        if language not in SUPPORTED_ADAPTER_LANGUAGES:
            return x
        
        if language_ids is None:
            if language is None:
                language = self.default_language
            
            batch_size = x.shape[0]
            lang_id = self.language_to_id[language]
            language_ids = torch.full((batch_size,), lang_id, device=x.device)
        
        # Get language embeddings
        lang_embeds = self.language_embed(language_ids)  # [batch, dim]
        
        # Apply language-specific adapters
        outputs = []
        for i, lang_id in enumerate(language_ids):
            lang_key = list(self.language_to_id.keys())[lang_id.item()]
            adapter_output = self.language_adapters[lang_key](x[i:i+1])
            outputs.append(adapter_output)
        
        adapted_x = torch.cat(outputs, dim=0)
        
        # Add language conditioning
        lang_embeds = lang_embeds.unsqueeze(1)  # [batch, 1, dim]
        adapted_x = adapted_x + lang_embeds
        
        return adapted_x


class AttentionAdapter(nn.Module):
    """
    Adapter specifically for attention layers with separate Q, K, V adapters.
    """
    
    def __init__(self, dim: int, num_heads: int, languages: Dict[str, Dict]):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Separate adapters for Q, K, V projections
        self.q_adapters = MultiLanguageAdapter(dim, languages)
        self.k_adapters = MultiLanguageAdapter(dim, languages)
        self.v_adapters = MultiLanguageAdapter(dim, languages)
        self.out_adapters = MultiLanguageAdapter(dim, languages)
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor,
        language: Optional[str] = None,
        language_ids: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply adapters to attention projections."""
        if language not in SUPPORTED_ADAPTER_LANGUAGES:
            return q, k, v, out
        
        adapted_q = self.q_adapters(q, language, language_ids)
        adapted_k = self.k_adapters(k, language, language_ids)
        adapted_v = self.v_adapters(v, language, language_ids)
        adapted_out = self.out_adapters(out, language, language_ids)
        
        return adapted_q, adapted_k, adapted_v, adapted_out


class TextEmbeddingAdapter(nn.Module):
    """
    Adapter for text embeddings to support different languages.
    """
    
    def __init__(
        self,
        base_vocab_size: int,
        extended_vocab_sizes: Dict[str, int],
        embed_dim: int,
        default_language: str = "en"
    ):
        super().__init__()
        self.base_vocab_size = base_vocab_size
        self.embed_dim = embed_dim
        self.default_language = default_language
        
        # Language-specific embedding extensions
        self.language_embeddings = nn.ModuleDict()
        for lang, vocab_size in extended_vocab_sizes.items():
            if vocab_size > base_vocab_size:
                # Only create extension for languages with larger vocabularies
                extension_size = vocab_size - base_vocab_size
                self.language_embeddings[lang] = nn.Embedding(extension_size, embed_dim)
                
                # Initialize with small values
                nn.init.normal_(self.language_embeddings[lang].weight, std=0.02)
    
    def forward(
        self,
        base_embeddings: torch.Tensor,
        token_ids: torch.Tensor,
        language: Optional[str] = None
    ) -> torch.Tensor:
        """
        Extend base embeddings with language-specific tokens.
        
        Args:
            base_embeddings: Base token embeddings [batch, seq_len, dim]
            token_ids: Token IDs [batch, seq_len]
            language: Target language
        """
        if language not in SUPPORTED_ADAPTER_LANGUAGES:
            return base_embeddings
        
        if language is None or language not in self.language_embeddings:
            return base_embeddings
        
        # Find tokens that exceed base vocabulary
        extended_mask = token_ids >= self.base_vocab_size
        
        if not extended_mask.any():
            return base_embeddings
        
        # Get extended embeddings for out-of-vocabulary tokens
        extended_token_ids = token_ids - self.base_vocab_size
        extended_token_ids = extended_token_ids.clamp(min=0)
        
        extended_embeddings = self.language_embeddings[language](extended_token_ids)
        
        # Combine base and extended embeddings
        output = base_embeddings.clone()
        output[extended_mask] = extended_embeddings[extended_mask]
        
        return output


class AdapterConfig:
    """Configuration for adapter setup."""
    
    def __init__(
        self,
        languages: Dict[str, Dict] = None,
        default_language: str = "en",
        adapter_type: str = "lora",
        rank: int = 16,
        alpha: float = 16.0,
        bottleneck_dim: int = 64,
        dropout: float = 0.1,
        enable_text_adapter: bool = True,
        enable_attention_adapter: bool = True,
        enable_feedforward_adapter: bool = True
    ):
        self.languages = languages or {
            "en": {"adapter_type": adapter_type, "rank": rank, "alpha": alpha, "dropout": dropout},
            "ro": {"adapter_type": adapter_type, "rank": rank, "alpha": alpha, "dropout": dropout}
        }
        self.default_language = default_language
        self.adapter_type = adapter_type
        self.rank = rank
        self.alpha = alpha
        self.bottleneck_dim = bottleneck_dim
        self.dropout = dropout
        self.enable_text_adapter = enable_text_adapter
        self.enable_attention_adapter = enable_attention_adapter
        self.enable_feedforward_adapter = enable_feedforward_adapter 