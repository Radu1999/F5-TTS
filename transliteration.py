import pandas as pd
import os
import torch
import torch.nn.functional as F
import soundfile as sf
import torchaudio
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
import json
from f5_tts.model.utils import list_str_to_idx
from f5_tts.infer.utils_infer import (
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text
)
from f5_tts.model import DiT


GENERATE_AUDIO = True


def load_romanian_model():
    """Load the Romanian F5-TTS model and vocoder"""
    print("Loading Romanian F5-TTS model...")

    # Model configuration for F5TTS_v1_Base (from setting.json)
    model_cfg = {
        'dim': 1024,
        'depth': 22,
        'heads': 16,
        'ff_mult': 2,
        'text_dim': 512,
        'conv_layers': 4
    }

    # Paths
    ckpt_path = "ckpts/ro_tts_lang_mod/model_last.pt"
    vocab_file = "data/ro_tts_pinyin/vocab.txt"

    # Load model
    model = load_model(
        model_cls=DiT,
        model_cfg=model_cfg,
        use_ema=False,
        ckpt_path=ckpt_path,
        vocab_file=vocab_file,
        mel_spec_type="vocos",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load vocoder
    vocoder = load_vocoder(
        vocoder_name="vocos",
        is_local=False,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Model and vocoder loaded successfully!")
    return model, vocoder


def create_output_directories():
    """Create directories for storing generated audio files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(f"transliteration_output_{timestamp}")
    
    reference_dir = base_dir / "reference_audio"
    transliterated_dir = base_dir / "transliterated_audio"
    
    reference_dir.mkdir(parents=True, exist_ok=True)
    transliterated_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Created output directories: {base_dir}")
    return reference_dir, transliterated_dir


def generate_reference_audio(model, vocoder, text, output_path, ref_audio_path=None, ref_text=None):
    """Generate reference audio for original Romanian text"""
    device = next(model.parameters()).device
    
    if ref_audio_path is None:
        # Use a real Romanian audio file as reference instead of silence
        print(f"  Generating reference audio for: {text[:50]}...")
        
        # Try to find an existing ground truth audio file to use as reference
        ground_truth_dir = Path("evaluation_outputs/romanian_f5tts_20250911_120739/ground_truth")
        if ground_truth_dir.exists():
            # Use the first available ground truth file as reference
            ref_files = list(ground_truth_dir.glob("*.wav"))
            if ref_files:
                ref_audio_path = ref_files[0]
                # Use a simple Romanian sentence as reference text
                ref_text = "Apoi mi-a fost și mai rău."
                print(f"    Using reference audio: {ref_audio_path}")
            else:
                # Fallback to silence if no files found
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    dummy_audio = torch.zeros(1, 16000)
                    sf.write(temp_file.name, dummy_audio.squeeze().numpy(), 16000)
                    ref_audio_path = temp_file.name
                    ref_text = text
        else:
            # Fallback to silence if directory doesn't exist
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                dummy_audio = torch.zeros(1, 16000)
                sf.write(temp_file.name, dummy_audio.squeeze().numpy(), 16000)
                ref_audio_path = temp_file.name
            ref_text = text
    
    try:
        ref_audio, ref_text_processed = preprocess_ref_audio_text(str(ref_audio_path), ref_text)
        
        generated_audio, sample_rate, _ = infer_process(
            ref_audio=ref_audio,
            ref_text=ref_text_processed,
            gen_text=text,
            model_obj=model,
            vocoder=vocoder,
            speed=0.95,
            device=device,
            show_info=print,
            cross_fade_duration=0.15,
            nfe_step=32,
            cfg_strength=2.0
        )
    finally:
        # Clean up temporary file if we created one
        if ref_audio_path and str(ref_audio_path).startswith(tempfile.gettempdir()):
            os.unlink(ref_audio_path)
    
    # Save the generated audio
    sf.write(output_path, generated_audio, sample_rate)
    print(f"    Saved reference audio: {output_path}")
    return str(output_path)


def generate_transliterated_audio(model, vocoder, transliterated_text, output_path, ref_audio_path=None, ref_text=None):
    """Generate audio for transliterated text"""
    device = next(model.parameters()).device
    
    print(f"  Generating transliterated audio for: {transliterated_text[:50]}...")
    
    if ref_audio_path is None:
        # Use the same Romanian reference audio as for reference generation
        ground_truth_dir = Path("evaluation_outputs/romanian_f5tts_20250911_120739/ground_truth")
        if ground_truth_dir.exists():
            # Use the first available ground truth file as reference
            ref_files = list(ground_truth_dir.glob("*.wav"))
            if ref_files:
                ref_audio_path = ref_files[0]
                # Use a simple Romanian sentence as reference text
                ref_text = "Apoi mi-a fost și mai rău."
                print(f"    Using reference audio: {ref_audio_path}")
            else:
                # Fallback to silence if no files found
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    dummy_audio = torch.zeros(1, 16000)
                    sf.write(temp_file.name, dummy_audio.squeeze().numpy(), 16000)
                    ref_audio_path = temp_file.name
                    ref_text = transliterated_text
        else:
            # Fallback to silence if directory doesn't exist
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                dummy_audio = torch.zeros(1, 16000)
                sf.write(temp_file.name, dummy_audio.squeeze().numpy(), 16000)
                ref_audio_path = temp_file.name
            ref_text = transliterated_text
    
    try:
        ref_audio, ref_text_processed = preprocess_ref_audio_text(str(ref_audio_path), ref_text)
        
        generated_audio, sample_rate, _ = infer_process(
            ref_audio=ref_audio,
            ref_text=ref_text_processed,
            gen_text=transliterated_text,
            model_obj=model,
            vocoder=vocoder,
            speed=0.95,
            device=device,
            show_info=print,
            cross_fade_duration=0.15,
            nfe_step=32,
            cfg_strength=2.0
        )
    finally:
        # Clean up temporary file if we created one
        if ref_audio_path and str(ref_audio_path).startswith(tempfile.gettempdir()):
            os.unlink(ref_audio_path)
    
    # Save the generated audio
    sf.write(output_path, generated_audio, sample_rate)
    print(f"    Saved transliterated audio: {output_path}")
    return str(output_path)

model, vocoder = load_romanian_model()

# Create output directories for audio files (only if audio generation is enabled)
if GENERATE_AUDIO:
    reference_dir, transliterated_dir = create_output_directories()

sentences = [
    "Apoi mi-a fost și mai rău."
]

for sentence_idx, sentence in enumerate(sentences):
    text = list_str_to_idx([sentence], model.vocab_char_map).to('cuda')
    ro_lang_emb, _, _ = model.transformer.language_module(text, seq_len=len(sentence))
    ro_lang_emb = model.transformer.text_embed(text, seq_len=text.shape[1], text_embed=ro_lang_emb)

    print(f"\nOriginal text indices: {text}")
    print(f"Original characters: {[model.vocab_char_map[k] for k in sentence if k in model.vocab_char_map]}")
    
    # Check what single-byte characters are available
    if hasattr(model, 'vocab_char_map'):
        idx_to_char = {v: k for k, v in model.vocab_char_map.items()}
        single_byte_chars = []
        for idx, char in idx_to_char.items():
            if len(char) == 1 and len(char.encode('utf-8')) == 1:
                single_byte_chars.append((idx, char))
        print(f"Available single-byte characters: {single_byte_chars[:20]}...")  # Show first 20
    
    transliteration = ""
    for ro_emb in ro_lang_emb.squeeze():
        # Get text embeddings weight matrix

        similarities = []
        # Compute embeddings and cosine similarity
        for ch in model.vocab_char_map.keys():
            if ch == '~':
                continue
            text = list_str_to_idx([ch], model.vocab_char_map).to('cuda')  # [1, seq_len]
            text_embed = model.transformer.text_embed(text, seq_len=text.shape[1])  # [1, seq_len, embed_dim]
            text_vec = text_embed[:, 0, :]  # take the first token embedding -> [1, embed_dim]
            sim = -torch.linalg.norm(text_vec - ro_emb.unsqueeze(0), dim=-1)
            similarities.append(sim.squeeze(0))  # scalar

        # Stack similarities into a single tensor
        similarities = torch.stack(similarities)  # shape [num_chars]
        top_k = 5
        top_values, top_indices = torch.topk(similarities, k=min(top_k, similarities.shape[0]))
        most_similar_idx = top_indices[0].item()

        if hasattr(model, 'vocab_char_map'):
            # Create reverse mapping from index to character
            idx_to_char = {v: k for k, v in model.vocab_char_map.items()}
            similar_char = idx_to_char.get(most_similar_idx, f"<idx_{most_similar_idx}>")
            top_matches = [(idx_to_char.get(i.item(), f"<idx_{i.item()}>"), v.item()) for i, v in zip(top_indices, top_values)]
        else:
            similar_char = f"<idx_{most_similar_idx}>"
            top_matches = [(f"<idx_{i.item()}>", v.item()) for i, v in zip(top_indices, top_values)]

        transliteration += similar_char

        print(f"Cosine top matches: {[(ch, round(score, 4)) for ch, score in top_matches]}")
    
    print(f"Original: {sentence}")
    print(f"Transliteration: {transliteration}")
    print("-" * 50)
    
    # Generate audio files (disabled)
    if GENERATE_AUDIO:
        try:
            # Create filenames
            safe_sentence = "".join(c for c in sentence[:30] if c.isalnum() or c in (' ', '_')).rstrip()
            safe_sentence = safe_sentence.replace(' ', '_')

            reference_filename = f"ref_{sentence_idx+1:02d}_{safe_sentence}.wav"
            transliterated_filename = f"trans_{sentence_idx+1:02d}_{safe_sentence}.wav"

            reference_path = reference_dir / reference_filename
            transliterated_path = transliterated_dir / transliterated_filename

            # Generate reference audio (original Romanian text)
            generate_reference_audio(model, vocoder, sentence, reference_path)

            # Generate transliterated audio
            generate_transliterated_audio(model, vocoder, transliteration, transliterated_path)

            print(f"Audio files generated:")
            print(f"  Reference: {reference_path}")
            print(f"  Transliterated: {transliterated_path}")

        except Exception as e:
            print(f"Error generating audio for sentence {sentence_idx+1}: {e}")
    else:
        print("Skipping audio generation (cosine similarity only).")
    
    print("=" * 50)
