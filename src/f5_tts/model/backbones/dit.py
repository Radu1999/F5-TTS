"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from x_transformers.x_transformers import RotaryEmbedding
from transformers import AutoModel, AutoTokenizer
from vector_quantize_pytorch import ResidualSimVQ, VectorQuantize, ResidualVQ, SimVQ
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
import os

from f5_tts.model.modules import (
    AdaLayerNorm_Final,
    ConvNeXtV2Block,
    ConvPositionEmbedding,
    DiTBlock,
    TimestepEmbedding,
    get_pos_embed_indices,
    precompute_freqs_cis,
    VQEmbedding
)

from f5_tts.model.utils import (
    list_str_to_idx,
    list_str_to_tensor,
    exists
)


class LanguageModule(nn.Module):
    def __init__(self, text_num_embeds=256, text_dim=None, conv_mult=2, conv_layers=4, vocab_char_map=None,
                 mask_padding=True):
        super().__init__()
        self.vq_layer = None
        self.codebook = None
        self.vocab_char_map = vocab_char_map
        self.text_blocks = nn.Sequential(
            *[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)]
        )
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)
        self.mask_padding = mask_padding
        self.pre_proj = None
        self.residual_vq = None

    def forward(self, text: int["b nt"], seq_len, drop_text=False, inference=False,
                step=None):  # noqa: F722
        if isinstance(text, list):
            text = list_str_to_idx(text, self.vocab_char_map).to('cuda')
        text = text + 1  # use 0 as filler token. preprocess of batch pad -1, see list_str_to_idx()
        text = text[:, :seq_len]  # curtail if character tokens are more than the mel spec tokens
        batch, text_len = text.shape[0], text.shape[1]
        source_text = F.pad(text, (0, seq_len - text_len), value=0)

        if self.mask_padding:
            text_mask = source_text == 0

        text = self.text_embed(source_text)
        text = text.masked_fill(text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1)), 0.0)
        for block in self.text_blocks:
            text = block(text)
            text = text.masked_fill(text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1)), 0.0)

        z_q, encoding_indices, loss = self.residual_vq(text)
        z_q = z_q.masked_fill(text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1)), 0.0)

        return z_q, loss.mean(), encoding_indices

    def build_vq(self, text_embed: nn.Embedding):
        self.residual_vq = ResidualVQ(
            dim=text_embed.weight.data.shape[1],
            codebook_size=32,
            num_quantizers=4,
        ).to('cuda')


    def visualize_text_embed_weights(self, weights_data):
        """
        Visualize text embedding weights with multiple visualization types
        Args:
            weights_data: tensor of shape (vocab_size, embedding_dim)
        """
        # Convert to numpy for visualization
        weights_np = weights_data.detach().cpu().numpy()
        vocab_size, embedding_dim = weights_np.shape
        
        # Create output directory for visualizations
        os.makedirs("text_embed_visualizations", exist_ok=True)
        
        # Set up the visualization style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Weight Distribution Histogram
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 3, 1)
        plt.hist(weights_np.flatten(), bins=50, alpha=0.7, density=True)
        plt.title('Distribution of Embedding Weights')
        plt.xlabel('Weight Value')
        plt.ylabel('Density')
        plt.grid(True, alpha=0.3)
        
        # 2. Heatmap of embedding weights (sampled if too large)
        plt.subplot(2, 3, 2)
        # Sample embeddings if vocabulary is too large for visualization
        max_vocab_display = 100
        if vocab_size > max_vocab_display:
            indices = np.linspace(0, vocab_size-1, max_vocab_display, dtype=int)
            weights_sample = weights_np[indices]
            title_suffix = f" (sampled {max_vocab_display}/{vocab_size})"
        else:
            weights_sample = weights_np
            title_suffix = ""
        
        sns.heatmap(weights_sample, cmap='RdBu_r', center=0, 
                   cbar_kws={'label': 'Weight Value'})
        plt.title(f'Embedding Weights Heatmap{title_suffix}')
        plt.xlabel('Embedding Dimension')
        plt.ylabel('Vocabulary Index')
        
        # 3. Statistics per embedding dimension
        plt.subplot(2, 3, 3)
        dim_means = np.mean(weights_np, axis=0)
        dim_stds = np.std(weights_np, axis=0)
        x_dims = np.arange(embedding_dim)
        
        plt.plot(x_dims, dim_means, label='Mean', alpha=0.8)
        plt.fill_between(x_dims, dim_means - dim_stds, dim_means + dim_stds, 
                        alpha=0.3, label='±1 std')
        plt.title('Statistics per Embedding Dimension')
        plt.xlabel('Dimension Index')
        plt.ylabel('Weight Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. L2 norms of embeddings
        plt.subplot(2, 3, 4)
        l2_norms = np.linalg.norm(weights_np, axis=1)
        plt.hist(l2_norms, bins=30, alpha=0.7)
        plt.title('L2 Norms of Embeddings')
        plt.xlabel('L2 Norm')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # 5. PCA Visualization (if embedding_dim > 2)
        plt.subplot(2, 3, 5)
        if embedding_dim > 2:
            pca = PCA(n_components=2)
            weights_pca = pca.fit_transform(weights_np)
            plt.scatter(weights_pca[:, 0], weights_pca[:, 1], alpha=0.6, s=20)
            plt.title(f'PCA Visualization\n(Explained variance: {pca.explained_variance_ratio_.sum():.3f})')
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
        else:
            plt.scatter(weights_np[:, 0], weights_np[:, 1] if embedding_dim > 1 else np.zeros_like(weights_np[:, 0]))
            plt.title('2D Embedding Visualization')
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2' if embedding_dim > 1 else 'Zero')
        plt.grid(True, alpha=0.3)
        
        # 6. Summary statistics
        plt.subplot(2, 3, 6)
        stats_text = f"""
        Embedding Statistics:
        
        Shape: {vocab_size} × {embedding_dim}
        Mean: {np.mean(weights_np):.4f}
        Std: {np.std(weights_np):.4f}
        Min: {np.min(weights_np):.4f}
        Max: {np.max(weights_np):.4f}
        
        Per-token L2 norm:
        Mean: {np.mean(l2_norms):.4f}
        Std: {np.std(l2_norms):.4f}
        """
        plt.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        plt.axis('off')
        plt.title('Summary Statistics')
        
        plt.tight_layout()
        plt.savefig('text_embed_visualizations/text_embed_weights_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.savefig('text_embed_visualizations/text_embed_weights_analysis.pdf', 
                   bbox_inches='tight')
        
        print(f"✅ Text embedding visualization saved to 'text_embed_visualizations/' directory")
        print(f"   - Embedding shape: {vocab_size} × {embedding_dim}")
        print(f"   - Weight statistics: mean={np.mean(weights_np):.4f}, std={np.std(weights_np):.4f}")
        print(f"   - L2 norm statistics: mean={np.mean(l2_norms):.4f}, std={np.std(l2_norms):.4f}")
        
        # Optional: Close the plot to free memory
        plt.close()


class TextEmbedding(nn.Module):
    def __init__(self, text_num_embeds, text_dim, mask_padding=True, conv_layers=0, conv_mult=2):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)  # use 0 as filler token

        self.mask_padding = mask_padding  # mask filler and batch padding tokens or not

        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 4096  # ~44s of 24khz audio
            self.register_buffer("freqs_cis", precompute_freqs_cis(text_dim, self.precompute_max_pos), persistent=False)
            self.text_blocks = nn.Sequential(
                *[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)]
            )
        else:
            self.extra_modeling = False

    def forward(self, text: int["b nt"], seq_len, text_embed=None, drop_text=False):  # noqa: F722
        text = text + 1  # use 0 as filler token. preprocess of batch pad -1, see list_str_to_idx()
        text = text[:, :seq_len]  # curtail if character tokens are more than the mel spec tokens
        batch, text_len = text.shape[0], text.shape[1]

        text = F.pad(text, (0, seq_len - text_len), value=0)
        if drop_text:  # cfg for text
            text = torch.zeros_like(text)

        if self.mask_padding:
            text_mask = text == 0

        if text_embed is not None:
            text = text_embed
        else:
            text = self.text_embed(text)  # b n -> b n d

        # possible extra modeling
        if self.extra_modeling:
            # sinus pos emb
            batch_start = torch.zeros((batch,), dtype=torch.long)
            pos_idx = get_pos_embed_indices(batch_start, seq_len, max_pos=self.precompute_max_pos)
            text_pos_embed = self.freqs_cis[pos_idx]
            text = text + text_pos_embed

            # convnextv2 blocks
            if self.mask_padding:
                text = text.masked_fill(text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1)), 0.0)
                for block in self.text_blocks:
                    text = block(text)
                    text = text.masked_fill(text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1)), 0.0)
            else:
                text = self.text_blocks(text)

        return text


# noised input audio and context mixing embedding


class InputEmbedding(nn.Module):
    def __init__(self, mel_dim, text_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(mel_dim * 2 + text_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(self, x: float["b n d"], cond: float["b n d"], text_embed: float["b n d"],
                drop_audio_cond=False):  # noqa: F722
        if drop_audio_cond:  # cfg for cond audio
            cond = torch.zeros_like(cond)

        x = self.proj(torch.cat((x, cond, text_embed), dim=-1))
        x = self.conv_pos_embed(x) + x
        return x


# Transformer backbone using DiT blocks


class DiT(nn.Module):
    def __init__(
            self,
            *,
            dim,
            depth=8,
            heads=8,
            dim_head=64,
            dropout=0.1,
            ff_mult=4,
            mel_dim=100,
            text_num_embeds=256,
            text_dim=None,
            text_mask_padding=True,
            qk_norm=None,
            conv_layers=0,
            pe_attn_head=None,
            attn_backend="torch",  # "torch" | "flash_attn"
            attn_mask_enabled=False,
            long_skip_connection=False,
            checkpoint_activations=False,
    ):
        super().__init__()

        self.time_embed = TimestepEmbedding(dim)
        if text_dim is None:
            text_dim = mel_dim
        self.text_embed = TextEmbedding(
            text_num_embeds, text_dim, mask_padding=text_mask_padding, conv_layers=conv_layers
        )
        self.text_cond, self.text_uncond = None, None  # text cache
        self.input_embed = InputEmbedding(mel_dim, text_dim, dim)

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

        self.transformer_blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    ff_mult=ff_mult,
                    dropout=dropout,
                    qk_norm=qk_norm,
                    pe_attn_head=pe_attn_head,
                    attn_backend=attn_backend,
                    attn_mask_enabled=attn_mask_enabled,
                )
                for _ in range(depth)
            ]
        )
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None

        self.norm_out = AdaLayerNorm_Final(dim)  # final modulation
        self.proj_out = nn.Linear(dim, mel_dim)

        self.checkpoint_activations = checkpoint_activations

        self.initialize_weights()

    def initialize_weights(self):
        # Zero-out AdaLN layers in DiT blocks:
        for block in self.transformer_blocks:
            nn.init.constant_(block.attn_norm.linear.weight, 0)
            nn.init.constant_(block.attn_norm.linear.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    def ckpt_wrapper(self, module):
        # https://github.com/chuanyangjin/fast-DiT/blob/main/models.py
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs

        return ckpt_forward

    def get_input_embed(
            self,
            x,  # b n d
            cond,  # b n d
            text,  # b nt
            drop_audio_cond: bool = False,
            drop_text: bool = False,
            cache: bool = True,
            text_embed=None,
    ):
        seq_len = x.shape[1]
        if cache:
            if drop_text:
                if self.text_uncond is None:
                    self.text_uncond = self.text_embed(text, seq_len, drop_text=True, text_embed=text_embed)
                text_embed = self.text_uncond
            else:
                if self.text_cond is None:
                    self.text_cond = self.text_embed(text, seq_len, drop_text=False, text_embed=text_embed)
                text_embed = self.text_cond
        else:
            text_embed = self.text_embed(text, seq_len, drop_text=drop_text, text_embed=text_embed)

        x = self.input_embed(x, cond, text_embed, drop_audio_cond=drop_audio_cond)

        return x

    def clear_cache(self):
        self.text_cond, self.text_uncond = None, None

    def forward(
            self,
            x: float["b n d"],  # nosied input audio  # noqa: F722
            cond: float["b n d"],  # masked cond audio  # noqa: F722
            text: int["b nt"],  # text  # noqa: F722
            time: float["b"] | float[""],  # time step  # noqa: F821 F722
            mask: bool["b n"] | None = None,  # noqa: F722
            drop_audio_cond: bool = False,  # cfg for cond audio
            drop_text: bool = False,  # cfg for text
            cfg_infer: bool = False,  # cfg inference, pack cond & uncond forward
            cache: bool = False,
            text_embed=None,
    ):
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        # t: conditioning time, text: text, x: noised audio + cond audio + text
        t = self.time_embed(time)
        if cfg_infer:  # pack cond & uncond forward: b n d -> 2b n d
            x_cond = self.get_input_embed(x, cond, text, drop_audio_cond=False, drop_text=False, text_embed=text_embed,
                                          cache=cache)
            x_uncond = self.get_input_embed(x, cond, text, drop_audio_cond=True, drop_text=True, text_embed=text_embed,
                                            cache=cache)
            x = torch.cat((x_cond, x_uncond), dim=0)
            t = torch.cat((t, t), dim=0)
            mask = torch.cat((mask, mask), dim=0) if mask is not None else None
        else:
            x = self.get_input_embed(x, cond, text, drop_audio_cond=drop_audio_cond, drop_text=drop_text, cache=cache,
                                     text_embed=text_embed)

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        if self.long_skip_connection is not None:
            residual = x

        for block in self.transformer_blocks:
            if self.checkpoint_activations:
                # https://pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.checkpoint
                x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, t, mask, rope, use_reentrant=False)
            else:
                x = block(x, t, mask=mask, rope=rope)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t)
        output = self.proj_out(x)

        return output
