#!/usr/bin/env python3
"""
Inference script for F5-TTS with language adapters.
Supports Romanian and English with voice cloning capabilities.
"""

import argparse
import os
import tempfile
from pathlib import Path
from typing import Optional, Union

import torch
import torchaudio
from cached_path import cached_path

from f5_tts.api import F5TTS
from f5_tts.model import CFM, DiT
from f5_tts.model.adapters import AdapterConfig
from f5_tts.model.utils import get_tokenizer


class F5TTSAdapters(F5TTS):
    """
    F5-TTS with adapter support for multi-language synthesis.
    """
    
    def __init__(
        self,
        model: str = "F5TTS_Base",
        ckpt_file: str = "",
        adapter_ckpt_file: str = "",
        vocab_file: str = "",
        device: str = None,
        use_ema: bool = True,
        adapter_config: Optional[AdapterConfig] = None,
        languages: list = ["en", "ro"],
        default_language: str = "en",
    ):
        """
        Initialize F5-TTS with adapter support.
        
        Args:
            model: Model type
            ckpt_file: Path to base model checkpoint
            adapter_ckpt_file: Path to adapter checkpoint
            vocab_file: Path to vocabulary file
            device: Device to use
            use_ema: Whether to use EMA weights
            adapter_config: Configuration for adapters
            languages: Supported languages
            default_language: Default language
        """
        self.adapter_ckpt_file = adapter_ckpt_file
        self.adapter_config = adapter_config or AdapterConfig(
            languages={lang: {"adapter_type": "lora", "rank": 16, "alpha": 16.0, "dropout": 0.1} 
                      for lang in languages},
            default_language=default_language
        )
        self.languages = languages
        self.default_language = default_language
        
        # Initialize base F5TTS
        super().__init__(model, ckpt_file, vocab_file, device, use_ema)
        
        # Load adapter weights if provided
        if adapter_ckpt_file and os.path.exists(adapter_ckpt_file):
            self.load_adapter_weights(adapter_ckpt_file)
    
    def load_adapter_weights(self, adapter_ckpt_file: str):
        """Load adapter-specific weights."""
        print(f"Loading adapter weights from: {adapter_ckpt_file}")
        
        if adapter_ckpt_file.endswith('.safetensors'):
            from safetensors.torch import load_file
            adapter_dict = load_file(adapter_ckpt_file, device=self.device)
        else:
            checkpoint = torch.load(adapter_ckpt_file, map_location=self.device)
            adapter_dict = checkpoint.get("ema_model_state_dict", checkpoint)
        
        # Load only adapter parameters
        model_dict = self.model.state_dict()
        adapter_params = {}
        
        for k, v in adapter_dict.items():
            if any(adapter_name in k for adapter_name in [
                'language_adapters', 'attention_adapter', 'ff_adapter', 
                'text_embedding_adapter', 'language_embed'
            ]):
                if k in model_dict:
                    adapter_params[k] = v
        
        model_dict.update(adapter_params)
        self.model.load_state_dict(model_dict, strict=False)
        
        print(f"Loaded {len(adapter_params)} adapter parameters")
    
    def infer(
        self,
        ref_file: Union[str, Path],
        ref_text: str,
        gen_text: str,
        nfe_step: int = 32,
        cfg_strength: float = 2.0,
        sway_sampling_coef: float = -1.0,
        speed: float = 1.0,
        fix_duration: Optional[float] = None,
        remove_silence: bool = False,
        file_wave: Optional[Union[str, Path]] = None,
        seed: int = -1,
        language: str = None,  # New parameter for language selection
    ):
        """
        Perform TTS inference with language adapter support.
        
        Args:
            ref_file: Reference audio file path
            ref_text: Reference text
            gen_text: Text to generate
            nfe_step: Number of sampling steps
            cfg_strength: Classifier-free guidance strength
            sway_sampling_coef: Sway sampling coefficient
            speed: Speech speed
            fix_duration: Fixed duration
            remove_silence: Whether to remove silence
            file_wave: Output file path
            seed: Random seed
            language: Target language ('en' or 'ro')
        """
        if language is None:
            language = self.default_language
        
        if language not in self.languages:
            print(f"Warning: Language '{language}' not supported. Using default '{self.default_language}'")
            language = self.default_language
        
        print(f"Generating speech in {language.upper()}")
        
        # Set random seed if specified
        if seed != -1:
            torch.manual_seed(seed)
            self.seed = seed
        else:
            self.seed = torch.seed()
        
        # Load reference audio
        ref_audio, sr = torchaudio.load(ref_file)
        if ref_audio.shape[0] > 1:
            ref_audio = ref_audio.mean(dim=0, keepdim=True)
        
        # Resample if necessary
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            ref_audio = resampler(ref_audio)
        
        ref_audio = ref_audio.to(self.device)
        
        # Prepare text inputs based on language
        if language == "ro":
            # For Romanian, we might need special text processing
            ref_text = self.preprocess_romanian_text(ref_text)
            gen_text = self.preprocess_romanian_text(gen_text)
        
        # Calculate duration
        ref_text_len = len(ref_text.encode('utf-8'))
        gen_text_len = len(gen_text.encode('utf-8'))
        duration = ref_audio.shape[-1] // self.hop_length + int(ref_audio.shape[-1] // self.hop_length * gen_text_len / ref_text_len)
        
        if fix_duration is not None:
            duration = int(fix_duration * self.target_sample_rate // self.hop_length)
        
        # Generate audio with language information
        with torch.inference_mode():
            generated, _ = self.model.sample(
                cond=ref_audio,
                text=[gen_text],
                duration=duration,
                steps=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef if sway_sampling_coef != -1 else None,
                seed=seed if seed != -1 else None,
                use_epss=True,
                language=language,  # Pass language information
            )
        
        # Apply speed adjustment
        if speed != 1.0:
            generated = self.adjust_speed(generated, speed)
        
        # Remove silence if requested
        if remove_silence:
            generated = self.remove_silence_segments(generated)
        
        # Save to file if specified
        if file_wave:
            torchaudio.save(file_wave, generated.cpu(), self.target_sample_rate)
            print(f"Generated audio saved to: {file_wave}")
        
        return generated.cpu(), self.seed
    
    def preprocess_romanian_text(self, text: str) -> str:
        """
        Preprocess Romanian text for better TTS quality.
        
        Args:
            text: Input Romanian text
            
        Returns:
            Preprocessed text
        """
        # Basic Romanian text preprocessing
        # This can be expanded with more sophisticated processing
        
        # Normalize Romanian diacritics
        romanian_replacements = {
            'ş': 'ș',  # Replace old cedilla with comma below
            'ţ': 'ț',
            'Ş': 'Ș',
            'Ţ': 'Ț',
        }
        
        for old, new in romanian_replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def adjust_speed(self, audio: torch.Tensor, speed: float) -> torch.Tensor:
        """Adjust speech speed using time-stretching."""
        if speed == 1.0:
            return audio
        
        # Simple time-stretching by resampling
        # For better quality, you might want to use more sophisticated methods
        original_length = audio.shape[-1]
        new_length = int(original_length / speed)
        
        if new_length != original_length:
            audio = torch.nn.functional.interpolate(
                audio.unsqueeze(0), 
                size=new_length, 
                mode='linear', 
                align_corners=False
            ).squeeze(0)
        
        return audio
    
    def remove_silence_segments(self, audio: torch.Tensor, threshold: float = 0.01) -> torch.Tensor:
        """Remove silence segments from generated audio."""
        # Simple silence removal based on energy threshold
        energy = torch.sqrt(torch.mean(audio ** 2, dim=0))
        non_silent = energy > threshold
        
        if non_silent.any():
            audio = audio[:, non_silent]
        
        return audio


def main():
    parser = argparse.ArgumentParser(description="F5-TTS Inference with Language Adapters")
    
    parser.add_argument("--ref_audio", type=str, required=True,
                        help="Path to reference audio file")
    parser.add_argument("--ref_text", type=str, required=True,
                        help="Reference text")
    parser.add_argument("--gen_text", type=str, required=True,
                        help="Text to generate")
    parser.add_argument("--output", type=str, required=True,
                        help="Output audio file path")
    
    # Model configuration
    parser.add_argument("--base_model", type=str, default="F5TTS_Base",
                        help="Base model type")
    parser.add_argument("--base_ckpt", type=str, required=True,
                        help="Path to base model checkpoint")
    parser.add_argument("--adapter_ckpt", type=str, default="",
                        help="Path to adapter checkpoint")
    parser.add_argument("--vocab_file", type=str, default="",
                        help="Path to vocabulary file")
    
    # Language configuration
    parser.add_argument("--language", type=str, default="en", choices=["en", "ro"],
                        help="Target language for generation")
    parser.add_argument("--languages", type=str, nargs="+", default=["en", "ro"],
                        help="Supported languages")
    
    # Generation parameters
    parser.add_argument("--steps", type=int, default=32,
                        help="Number of sampling steps")
    parser.add_argument("--cfg_strength", type=float, default=2.0,
                        help="Classifier-free guidance strength")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Speech speed multiplier")
    parser.add_argument("--seed", type=int, default=-1,
                        help="Random seed (-1 for random)")
    parser.add_argument("--remove_silence", action="store_true",
                        help="Remove silence from generated audio")
    
    # Device configuration
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use for inference")
    parser.add_argument("--use_ema", action="store_true", default=True,
                        help="Use EMA weights")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Setup adapter configuration
    adapter_config = AdapterConfig(
        languages={lang: {"adapter_type": "lora", "rank": 16, "alpha": 16.0, "dropout": 0.1} 
                  for lang in args.languages},
        default_language="en"
    )
    
    # Initialize F5-TTS with adapters
    tts = F5TTSAdapters(
        model=args.base_model,
        ckpt_file=args.base_ckpt,
        adapter_ckpt_file=args.adapter_ckpt,
        vocab_file=args.vocab_file,
        device=device,
        use_ema=args.use_ema,
        adapter_config=adapter_config,
        languages=args.languages,
        default_language="en",
    )
    
    # Generate audio
    print(f"Generating audio with language: {args.language}")
    print(f"Reference: {args.ref_text}")
    print(f"Generate: {args.gen_text}")
    
    audio, seed = tts.infer(
        ref_file=args.ref_audio,
        ref_text=args.ref_text,
        gen_text=args.gen_text,
        nfe_step=args.steps,
        cfg_strength=args.cfg_strength,
        speed=args.speed,
        remove_silence=args.remove_silence,
        file_wave=args.output,
        seed=args.seed,
        language=args.language,
    )
    
    print(f"Audio generated successfully!")
    print(f"Output saved to: {args.output}")
    print(f"Used seed: {seed}")


if __name__ == "__main__":
    main() 