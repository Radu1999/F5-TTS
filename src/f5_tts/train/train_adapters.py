#!/usr/bin/env python3
"""
Training script for language adapters in F5-TTS.
This script enables multi-language support by training adapters for Romanian
while preserving English capabilities.
"""

import argparse
import os
import shutil
from importlib.resources import files
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from cached_path import cached_path
from torch.optim import AdamW
from torch.utils.data import DataLoader

from f5_tts.model import CFM, DiT, Trainer
from f5_tts.model.adapters import AdapterConfig
from f5_tts.model.dataset import load_dataset
from f5_tts.model.utils import get_tokenizer


def create_romanian_tokenizer(base_tokenizer_path: str, romanian_data_path: str) -> Dict:
    """
    Create an extended tokenizer for Romanian by analyzing the Romanian dataset.
    
    Args:
        base_tokenizer_path: Path to the base tokenizer
        romanian_data_path: Path to Romanian training data
        
    Returns:
        Extended vocabulary mapping
    """
    # Load base tokenizer
    base_vocab_char_map, base_vocab_size = get_tokenizer(base_tokenizer_path, "char")
    
    # Romanian-specific characters that might not be in the base vocabulary
    romanian_chars = [
        'ă', 'â', 'î', 'ș', 'ț',  # Romanian diacritics
        'Ă', 'Â', 'Î', 'Ș', 'Ț',  # Uppercase versions
    ]
    
    # Create extended vocabulary
    extended_vocab = base_vocab_char_map.copy()
    current_id = base_vocab_size
    
    for char in romanian_chars:
        if char not in extended_vocab:
            extended_vocab[char] = current_id
            current_id += 1
    
    print(f"Extended vocabulary size: {len(extended_vocab)} (added {len(extended_vocab) - base_vocab_size} Romanian characters)")
    
    return extended_vocab, len(extended_vocab)


def freeze_base_model(model: nn.Module, adapter_config: AdapterConfig) -> None:
    """
    Freeze the base model parameters, keeping only adapter parameters trainable.
    
    Args:
        model: The F5-TTS model
        adapter_config: Configuration for adapters
    """
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze adapter parameters
    for name, module in model.named_modules():
        if any(adapter_name in name for adapter_name in [
            'language_adapters', 'attention_adapter', 'ff_adapter', 
            'text_embedding_adapter', 'language_embed'
        ]):
            for param in module.parameters():
                param.requires_grad = True
            print(f"Unfrozen adapter module: {name}")
    
    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")


def setup_adapter_model(
    base_model_path: str,
    adapter_config: AdapterConfig,
    vocab_size: int,
    mel_dim: int = 100
) -> CFM:
    """
    Setup the F5-TTS model with adapters.
    
    Args:
        base_model_path: Path to the base pre-trained model
        adapter_config: Configuration for adapters
        vocab_size: Size of the extended vocabulary
        mel_dim: Mel spectrogram dimension
        
    Returns:
        CFM model with adapters
    """
    # Model configuration (F5TTS_Base)
    model_cfg = dict(
        dim=1024,
        depth=22,
        heads=16,
        ff_mult=2,
        text_dim=512,
        conv_layers=4,
        adapter_config=adapter_config,
    )
    
    # Mel spectrogram configuration
    mel_spec_kwargs = dict(
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=mel_dim,
        target_sample_rate=24000,
        mel_spec_type="vocos",
    )
    
    # Create model with adapters
    model = CFM(
        transformer=DiT(
            **model_cfg,
            text_num_embeds=vocab_size,
            mel_dim=mel_dim
        ),
        mel_spec_kwargs=mel_spec_kwargs,
        adapter_config=adapter_config,
    )
    
    # Load base model weights
    if os.path.exists(base_model_path):
        print(f"Loading base model from: {base_model_path}")
        
        if base_model_path.endswith('.safetensors'):
            from safetensors.torch import load_file
            checkpoint = {"ema_model_state_dict": load_file(base_model_path, device="cpu")}
        else:
            checkpoint = torch.load(base_model_path, map_location="cpu")
        
        # Load weights, ignoring adapter-specific parameters
        model_dict = model.state_dict()
        pretrained_dict = checkpoint.get("ema_model_state_dict", checkpoint)
        
        # Filter out adapter parameters and size mismatches
        filtered_dict = {}
        for k, v in pretrained_dict.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                if not any(adapter_name in k for adapter_name in [
                    'language_adapters', 'attention_adapter', 'ff_adapter', 
                    'text_embedding_adapter', 'language_embed'
                ]):
                    filtered_dict[k] = v
        
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict, strict=False)
        
        print(f"Loaded {len(filtered_dict)} parameters from base model")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train language adapters for F5-TTS")
    
    # Experiment configuration
    parser.add_argument("--exp_name", type=str, default="F5TTS_Adapters",
                        help="Name of the experiment")
    
    # Model and data configuration
    parser.add_argument("--base_model_path", type=str, default=None,
                        help="Path to the base pre-trained model (deprecated, use --pretrain)")
    parser.add_argument("--pretrain", type=str, default=None,
                        help="Path to the pre-trained model")
    parser.add_argument("--romanian_dataset", type=str, default=None,
                        help="Path to Romanian dataset")
    parser.add_argument("--english_dataset", type=str, default=None,
                        help="Path to English dataset (optional, for joint training)")
    parser.add_argument("--dataset_name", type=str, default="ro_tts",
                        help="Name of the dataset")
    parser.add_argument("--output_dir", type=str, default="./checkpoints",
                        help="Directory to save adapter checkpoints")
    parser.add_argument("--finetune", action="store_true",
                        help="Enable finetuning mode")
    parser.add_argument("--tokenizer", type=str, default="char", choices=["char", "pinyin"],
                        help="Tokenizer type to use")
    
    # Adapter configuration
    parser.add_argument("--adapter_type", type=str, default="bottleneck", choices=["lora", "bottleneck"],
                        help="Type of adapter to use")
    parser.add_argument("--adapter_rank", type=int, default=16,
                        help="Rank for LoRA adapters")
    parser.add_argument("--adapter_alpha", type=float, default=16.0,
                        help="Alpha scaling for LoRA adapters")
    parser.add_argument("--adapter_dropout", type=float, default=0.1,
                        help="Dropout rate for adapters")
    
    # Training configuration
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate for adapter training")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size per GPU (deprecated, use --batch_size_per_gpu)")
    parser.add_argument("--batch_size_per_gpu", type=int, default=16,
                        help="Batch size per GPU")
    parser.add_argument("--batch_size_type", type=str, default="sample", choices=["sample", "frame"],
                        help="Batch size type")
    parser.add_argument("--max_samples", type=int, default=64,
                        help="Maximum number of samples per batch")
    parser.add_argument("--grad_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Maximum gradient norm for clipping")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=1000,
                        help="Number of warmup steps (deprecated, use --num_warmup_updates)")
    parser.add_argument("--num_warmup_updates", type=int, default=1000,
                        help="Number of warmup updates")
    parser.add_argument("--save_steps", type=int, default=1000,
                        help="Save checkpoint every N steps (deprecated, use --save_per_updates)")
    parser.add_argument("--save_per_updates", type=int, default=1000,
                        help="Save checkpoint every N updates")
    parser.add_argument("--keep_last_n_checkpoints", type=int, default=3,
                        help="Keep last N checkpoints (-1 for all)")
    parser.add_argument("--last_per_updates", type=int, default=1000,
                        help="Save last checkpoint every N updates")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="Evaluate every N steps")
    
    # Language configuration
    parser.add_argument("--languages", type=str, nargs="+", default=["en", "ro"],
                        help="Languages to support")
    parser.add_argument("--default_language", type=str, default="en",
                        help="Default language")
    
    # Logging configuration
    parser.add_argument("--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb"],
                        help="Logger type")
    parser.add_argument("--log_samples", action="store_true",
                        help="Log sample outputs during training")
    
    # Other options
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["none", "fp16", "bf16"],
                        help="Mixed precision training")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use for training")
    
    args = parser.parse_args()
    
    # Handle deprecated arguments
    if args.pretrain is None and args.base_model_path is not None:
        args.pretrain = args.base_model_path
    
    if args.pretrain is None:
        raise ValueError("Either --pretrain or --base_model_path must be specified")
    
    # Use the more specific argument names
    batch_size_per_gpu = args.batch_size_per_gpu if args.batch_size_per_gpu != 16 else args.batch_size
    num_warmup_updates = args.num_warmup_updates if args.num_warmup_updates != 1000 else args.warmup_steps
    save_per_updates = args.save_per_updates if args.save_per_updates != 1000 else args.save_steps
    
    # Setup device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup tokenizer
    print(f"Setting up tokenizer: {args.tokenizer}")
    if args.tokenizer == "char":
        # For character-based tokenizer, extend for Romanian
        vocab_char_map, vocab_size = create_romanian_tokenizer(
            base_tokenizer_path=args.romanian_dataset or args.dataset_name,
            romanian_data_path=args.romanian_dataset or args.dataset_name
        )
    else:
        # For pinyin tokenizer, use standard approach
        vocab_char_map, vocab_size = get_tokenizer(args.dataset_name, args.tokenizer)
    
    # Setup adapter configuration
    adapter_config = AdapterConfig(
        languages={
            "en": {
                "adapter_type": args.adapter_type,
                "rank": args.adapter_rank,
                "alpha": args.adapter_alpha,
                "dropout": args.adapter_dropout
            },
            "ro": {
                "adapter_type": args.adapter_type,
                "rank": args.adapter_rank,
                "alpha": args.adapter_alpha,
                "dropout": args.adapter_dropout
            }
        },
        default_language=args.default_language,
        enable_text_adapter=True,
        enable_attention_adapter=True,
        enable_feedforward_adapter=True,
    )
    
    # Setup model
    print("Setting up model with adapters...")
    model = setup_adapter_model(
        base_model_path=args.pretrain,
        adapter_config=adapter_config,
        vocab_size=vocab_size
    )
    
    # Freeze base model parameters if finetuning
    if args.finetune:
        freeze_base_model(model, adapter_config)
    
    # Setup trainer
    trainer = Trainer(
        model,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        num_warmup_updates=num_warmup_updates,
        save_per_updates=save_per_updates,
        keep_last_n_checkpoints=args.keep_last_n_checkpoints,
        checkpoint_path=args.output_dir,
        batch_size_per_gpu=batch_size_per_gpu,
        batch_size_type=args.batch_size_type,
        max_samples=args.max_samples,
        grad_accumulation_steps=args.grad_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        logger=args.logger,
        wandb_project="F5-TTS-Romanian-Adapters",
        wandb_run_name=args.exp_name,
        log_samples=args.log_samples,
        last_per_updates=args.last_per_updates,
        mel_spec_type="vocos",
        is_local_vocoder=False,
        model_cfg_dict=vars(args),
    )
    
    # Load dataset
    print(f"Loading dataset: {args.dataset_name}")
    dataset_path = args.romanian_dataset or args.dataset_name
    
    train_dataset = load_dataset(
        dataset_path,
        tokenizer=args.tokenizer,
        mel_spec_kwargs=dict(
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_mel_channels=100,
            target_sample_rate=24000,
            mel_spec_type="vocos",
        )
    )
    
    # Add language information to dataset
    def add_language_info(batch):
        batch["language"] = "ro"  # Romanian language
        return batch
    
    # Start training
    print("Starting adapter training...")
    trainer.train(
        train_dataset,
        num_workers=args.num_workers,
        resumable_with_seed=42,
    )
    
    print("Training completed!")
    print(f"Adapters saved to: {args.output_dir}")


if __name__ == "__main__":
    main()