"""
Audio Generation Script for Full ro_tts_common Dataset

This script generates audio for the entire ro_tts_common dataset using either:
- F5-TTS Romanian model (voice cloning with reference audio)
- MMS-TTS-RON model (facebook/mms-tts-ron, direct text-to-speech)

The generated audio will be saved in an organized structure for later evaluation.
Use --model_type argument to choose between 'f5tts' (default) and 'mms'.
"""

import pandas as pd
import os
import torch
import soundfile as sf
import torchaudio
import shutil
from datetime import datetime
from pathlib import Path
import json
from tqdm import tqdm
import argparse
import sys
import re
import scipy.io.wavfile
from transformers import VitsModel, AutoTokenizer

from f5_tts.infer.utils_infer import (
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text
)
from f5_tts.model import DiT

# Set UTF-8 encoding for console output on Windows
if sys.platform.startswith('win'):
    import locale
    try:
        # Try to set UTF-8 locale
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except locale.Error:
        try:
            # Fallback to Romanian locale if available
            locale.setlocale(locale.LC_ALL, 'ro_RO.UTF-8')
        except locale.Error:
            pass
    
    # Set console to handle UTF-8
    try:
        import codecs
        import io
        # Reconfigure stdout and stderr for UTF-8
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

TARGET_SAMPLE_RATE = 24000


def normalize_romanian_text(text):
    """
    Normalize Romanian text for consistent processing
    - Replace cedilla (ş, ţ) with comma diacritics (ș, ț) 
    - Normalize other Romanian characters
    """
    if not text:
        return text
    
    # Dictionary of character replacements
    replacements = {
        # Cedilla to comma diacritics (preferred in Romanian)
        'ş': 'ș',  # s with cedilla -> s with comma below
        'Ş': 'Ș',  # S with cedilla -> S with comma below
        'ţ': 'ț',  # t with cedilla -> t with comma below
        'Ţ': 'Ț',  # T with cedilla -> T with comma below
        
        # Other normalizations if needed
        'ă': 'ă',  # a with breve (should already be correct)
        'Ă': 'Ă',  # A with breve
        'â': 'â',  # a with circumflex
        'Â': 'Â',  # A with circumflex
        'î': 'î',  # i with circumflex
        'Î': 'Î',  # I with circumflex
    }
    
    # Apply replacements
    normalized_text = text
    for old_char, new_char in replacements.items():
        normalized_text = normalized_text.replace(old_char, new_char)
    
    return normalized_text


def safe_print(text):
    """Safely print text with Romanian diacritics"""
    try:
        print(text)
    except UnicodeEncodeError:
        # Fallback: encode with error handling
        try:
            safe_text = str(text).encode('utf-8', errors='replace').decode('utf-8')
            print(safe_text)
        except Exception:
            # Last resort: ASCII-safe representation
            safe_text = repr(text)
            print(safe_text)


def load_romanian_model():
    """Load the Romanian F5-TTS model and vocoder"""
    print("Loading Romanian F5-TTS model...")

    model_cfg = {
        'dim': 1024,
        'depth': 22,
        'heads': 16,
        'ff_mult': 2,
        'text_dim': 512,
        'conv_layers': 4
    }

    ckpt_path = "ckpts/ro_tts_lang_mod/model_last.pt"
    vocab_file = "data/ro_tts_pinyin/vocab.txt"

    model = load_model(
        model_cls=DiT,
        model_cfg=model_cfg,
        use_ema=False,
        ckpt_path=ckpt_path,
        vocab_file=vocab_file,
        mel_spec_type="vocos",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    vocoder = load_vocoder(
        vocoder_name="vocos",
        is_local=False,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Model and vocoder loaded successfully!")
    return model, vocoder


def load_mms_romanian_model():
    """Load the MMS-TTS-RON model and tokenizer"""
    print("Loading MMS-TTS-RON model...")
    model = VitsModel.from_pretrained("facebook/mms-tts-ron")
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-ron")
    print("MMS-TTS-RON model loaded successfully!")
    return model, tokenizer


def load_metadata(metadata_path="data/ro_tts_common_pinyin/metadata.csv"):
    """Load the full ro_tts_common dataset metadata"""
    safe_print(f"Loading metadata from {metadata_path}...")
    
    # Load metadata - format: audio_path|text|client_id
    df = pd.read_csv(metadata_path, sep='|', names=['audio_path', 'text', 'client_id'], encoding='utf-8')
    
    # Normalize Romanian text
    safe_print("Normalizing Romanian text (cedilla -> comma diacritics)...")
    df['text'] = df['text'].apply(normalize_romanian_text)
    
    safe_print(f"Loaded {len(df)} samples from dataset")
    safe_print(f"Number of unique speakers: {df['client_id'].nunique()}")
    
    return df


def create_output_structure(base_output_dir="full_dataset_generation"):
    """Create organized folder structure for generated audio"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_output_dir) / f"ro_tts_common_generated_{timestamp}"
    
    generated_dir = output_dir / "generated_audio"
    reference_dir = output_dir / "reference_audio"
    metadata_dir = output_dir / "metadata"
    
    for dir_path in [generated_dir, reference_dir, metadata_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Created output directory structure at: {output_dir}")
    return output_dir, generated_dir, reference_dir, metadata_dir


def get_audio_file_path(csv_filename, wavs_dir="data/ro_tts_common_pinyin/wavs"):
    """Convert CSV filename to full path"""
    return Path(wavs_dir) / csv_filename.replace('wavs/', '')


def get_reference_audio_from_metadata(metadata_line, wavs_dir="data/ro_tts_common_pinyin/wavs"):
    """Parse a metadata line and return the audio path and text
    
    Format: hash;filename;age;gender;text
    Example: 477ecca08cd76f5431d459a2e73d2509c5249b1719dfb5415b40b0e79fc29488a2d6f046697ee436002d356ee039bff0fc3758c6e31626299d3e05fce0110ecc;common_voice_ro_25954347.mp3;twenties;male;Apoi mi-a fost și mai rău.
    """
    parts = metadata_line.strip().split(';')
    if len(parts) >= 5:
        filename = parts[1]
        text = parts[4]
        # Convert .mp3 to .wav if needed
        if filename.endswith('.mp3'):
            filename = filename.replace('.mp3', '.wav')
        audio_path = Path(wavs_dir) / filename
        return str(audio_path), text
    else:
        raise ValueError(f"Invalid metadata format: {metadata_line}")


def generate_single_audio(model, vocoder, row, generated_dir, reference_dir, idx, single_ref_audio=None, single_ref_text=None, model_type="f5tts"):
    """Generate audio for a single sample using infer_process or MMS-TTS-RON"""
    
    try:
        if model_type == "mms":
            return generate_single_audio_mms(model, vocoder, row, generated_dir, reference_dir, idx, single_ref_audio, single_ref_text)
        else:
            return generate_single_audio_f5tts(model, vocoder, row, generated_dir, reference_dir, idx, single_ref_audio, single_ref_text)
        
    except Exception as e:
        safe_print(f"  Error processing sample {idx}: {e}")
        return None, None


def generate_single_audio_f5tts(model, vocoder, row, generated_dir, reference_dir, idx, single_ref_audio=None, single_ref_text=None):
    """Generate audio for a single sample using F5-TTS infer_process"""
    device = next(model.parameters()).device
    
    if False and single_ref_audio and single_ref_text:
        # Use single reference audio for all generations
        ref_audio_path = single_ref_audio
        ref_text = single_ref_text
        gen_text = row['text']  # Generate the target text
        
        # Copy reference audio to reference directory once
        ref_dest_path = reference_dir / "single_reference.wav"
        if not ref_dest_path.exists():  # Only copy once
            shutil.copy2(single_ref_audio, ref_dest_path)
        
        sample_ref_audio_path = str(ref_dest_path)
        original_audio_path = single_ref_audio
    else:
        # Original behavior: use each sample's own audio as reference
        ref_audio_path = get_audio_file_path(row['audio_path'])
        
        if not ref_audio_path.exists():
            safe_print(f"  Warning: Reference audio not found: {ref_audio_path}")
            return None, None
        
        # Copy reference audio
        ref_filename = f"ref_{idx:06d}.wav"
        ref_dest_path = reference_dir / ref_filename
        shutil.copy2(ref_audio_path, ref_dest_path)
        
        # Self-reconstruction: use same text for reference and generation
        ref_text = row['text']
        gen_text = row['text']
        
        sample_ref_audio_path = str(ref_dest_path)
        original_audio_path = str(ref_audio_path)
    
    # Preprocess reference audio and text
    ref_audio_processed, ref_text_processed = preprocess_ref_audio_text(
        str(ref_audio_path), 
        ref_text
    )
    
    # Generate audio using infer_process
    generated_audio, sample_rate, _ = infer_process(
        ref_audio=ref_audio_processed,
        ref_text=ref_text_processed,
        gen_text=gen_text,
        model_obj=model,
        vocoder=vocoder,
        speed=1.0,
        device=device,
        show_info=None,  # Disable verbose output
        cross_fade_duration=0.15,
        nfe_step=32,
        cfg_strength=2.0
    )
    
    # Save generated audio
    gen_filename = f"gen_{idx:06d}.wav"
    gen_output_path = generated_dir / gen_filename
    sf.write(gen_output_path, generated_audio, sample_rate)
    
    # Create metadata entries
    generated_file = {
        'sample_id': idx,
        'original_audio_path': original_audio_path,
        'reference_audio_path': sample_ref_audio_path,
        'generated_audio_path': str(gen_output_path),
        'text': row['text'],
        'client_id': row['client_id'],
        'sample_rate': sample_rate,
        'model': 'f5tts'
    }
    
    reference_file = {
        'sample_id': idx,
        'reference_audio_path': sample_ref_audio_path,
        'text': row['text'],
        'client_id': row['client_id']
    }
    
    return generated_file, reference_file


def generate_single_audio_mms(model, tokenizer, row, generated_dir, reference_dir, idx, single_ref_audio=None, single_ref_text=None):
    """Generate audio for a single sample using MMS-TTS-RON"""
    
    # For MMS-TTS-RON, we don't need reference audio as it's not a voice cloning model
    # We'll still copy the original audio as reference for evaluation purposes
    
    # Get reference audio path for copying to reference directory
    if single_ref_audio:
        ref_audio_path = single_ref_audio
        original_audio_path = single_ref_audio
        # Copy reference audio to reference directory once
        ref_dest_path = reference_dir / "single_reference.wav"
        if not ref_dest_path.exists():  # Only copy once
            shutil.copy2(single_ref_audio, ref_dest_path)
        sample_ref_audio_path = str(ref_dest_path)
    else:
        ref_audio_path = get_audio_file_path(row['audio_path'])
        
        if not ref_audio_path.exists():
            safe_print(f"  Warning: Reference audio not found: {ref_audio_path}")
            return None, None
        
        # Copy reference audio
        ref_filename = f"ref_{idx:06d}.wav"
        ref_dest_path = reference_dir / ref_filename
        shutil.copy2(ref_audio_path, ref_dest_path)
        
        sample_ref_audio_path = str(ref_dest_path)
        original_audio_path = str(ref_audio_path)
    
    # Generate text to speech using MMS-TTS-RON
    gen_text = row['text']
    inputs = tokenizer(gen_text, return_tensors="pt")
    
    with torch.no_grad():
        output = model(**inputs).waveform
    
    # Save generated audio
    gen_filename = f"gen_{idx:06d}.wav"
    gen_output_path = generated_dir / gen_filename
    
    scipy.io.wavfile.write(
        str(gen_output_path), 
        rate=model.config.sampling_rate, 
        data=output.squeeze().cpu().numpy()
    )
    
    # Create metadata entries
    generated_file = {
        'sample_id': idx,
        'original_audio_path': original_audio_path,
        'reference_audio_path': sample_ref_audio_path,
        'generated_audio_path': str(gen_output_path),
        'text': row['text'],
        'client_id': row['client_id'],
        'sample_rate': model.config.sampling_rate,
        'model': 'facebook/mms-tts-ron'
    }
    
    reference_file = {
        'sample_id': idx,
        'reference_audio_path': sample_ref_audio_path,
        'text': row['text'],
        'client_id': row['client_id']
    }
    
    return generated_file, reference_file


def generate_audio_batch(model, vocoder, df_batch, generated_dir, reference_dir, batch_idx, total_batches, single_ref_audio=None, single_ref_text=None, model_type="f5tts"):
    """Generate audio for a batch of samples using individual inference
    
    Args:
        single_ref_audio: Optional single reference audio path to use for all generations
        single_ref_text: Reference text corresponding to the single reference audio
        model_type: Type of model to use ("f5tts" or "mms")
    """
    generated_files = []
    reference_files = []
    
    print(f"\nProcessing batch {batch_idx + 1}/{total_batches} ({len(df_batch)} samples)")
    
    # Process each sample individually
    for idx, row in tqdm(df_batch.iterrows(), total=len(df_batch), desc=f"Batch {batch_idx + 1}"):
        generated_file, reference_file = generate_single_audio(
            model, vocoder, row, generated_dir, reference_dir, idx, 
            single_ref_audio, single_ref_text, model_type
        )
        
        if generated_file and reference_file:
            generated_files.append(generated_file)
            reference_files.append(reference_file)
    
    safe_print(f"  Batch {batch_idx + 1} completed: {len(generated_files)} successful generations")
    return generated_files, reference_files


def save_generation_metadata(output_dir, metadata_dir, generated_files, reference_files, total_samples, model_type="f5tts"):
    """Save generation metadata and information"""
    
    # Determine model name and config based on type
    if model_type == "mms":
        model_name = 'facebook/mms-tts-ron'
        generation_method = 'text_to_speech'
        model_config = {
            'model_type': 'VITS',
            'sampling_rate': 16000  # MMS-TTS-RON default
        }
    else:
        model_name = 'romanian_f5tts'
        generation_method = 'self_reconstruction'
        model_config = {
            'speed': 1.0,
            'cross_fade_duration': 0.15,
            'nfe_step': 32,
            'cfg_strength': 2.0
        }
    
    metadata = {
        'generation_info': {
            'model_name': model_name,
            'model_type': model_type,
            'dataset': 'ro_tts_common',
            'timestamp': datetime.now().isoformat(),
            'total_samples_attempted': total_samples,
            'successful_generations': len(generated_files),
            'target_sample_rate': TARGET_SAMPLE_RATE,
            'generation_method': generation_method,
            'model_config': model_config
        },
        'generated_samples': generated_files,
        'reference_samples': reference_files
    }
    
    # Save JSON metadata
    metadata_file = metadata_dir / "generation_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # Save CSV files for easy analysis
    if generated_files:
        generated_df = pd.DataFrame(generated_files)
        generated_df.to_csv(metadata_dir / "generated_samples.csv", index=False)
    
    if reference_files:
        reference_df = pd.DataFrame(reference_files)
        reference_df.to_csv(metadata_dir / "reference_samples.csv", index=False)
    
    # Save summary
    summary_file = metadata_dir / "generation_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"Audio Generation Summary\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Dataset: ro_tts_common\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Model Type: {model_type}\n")
        f.write(f"Generation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Output Directory: {output_dir}\n\n")
        f.write(f"Statistics:\n")
        f.write(f"  Total Samples Attempted: {total_samples}\n")
        f.write(f"  Successful Generations: {len(generated_files)}\n")
        f.write(f"  Success Rate: {len(generated_files)/total_samples*100:.2f}%\n")
        f.write(f"  Target Sample Rate: {TARGET_SAMPLE_RATE}Hz\n\n")
        f.write(f"Generation Method: {generation_method}\n")
        if model_type == "mms":
            f.write(f"  Note: MMS-TTS-RON generates speech directly from text (no reference audio needed)\n\n")
        else:
            f.write(f"  Note: F5-TTS uses reference audio for voice cloning/self-reconstruction\n\n")
        f.write(f"Directory Structure:\n")
        f.write(f"  Generated Audio: {output_dir / 'generated_audio'}\n")
        f.write(f"  Reference Audio: {output_dir / 'reference_audio'}\n")
        f.write(f"  Metadata: {output_dir / 'metadata'}\n")
    
    safe_print(f"\nGeneration metadata saved to: {metadata_dir}")
    return metadata_file


def main():
    parser = argparse.ArgumentParser(description='Generate audio for full ro_tts_common dataset')
    parser.add_argument('--batch_size', type=int, default=100, 
                        help='Number of samples to process in each batch (default: 100, for progress tracking only)')
    parser.add_argument('--max_samples', type=int, default=1000,
                        help='Maximum number of samples to process (default: all)')
    parser.add_argument('--start_from', type=int, default=0,
                        help='Start processing from this sample index (default: 0)')
    parser.add_argument('--output_dir', type=str, default="full_dataset_generation",
                        help='Base output directory (default: full_dataset_generation)')
    parser.add_argument('--metadata_path', type=str, default="data/ro_tts_common_pinyin/metadata.csv",
                        help='Path to metadata CSV file')
    parser.add_argument('--single_ref_audio', type=str, default='data/ro_tts_common_pinyin/wavs/common_voice_ro_25954347.wav',
                        help='Path to single reference audio to use for all generations')
    parser.add_argument('--single_ref_text', type=str, default=None,
                        help='Reference text corresponding to the single reference audio')
    parser.add_argument('--model_type', type=str, choices=['f5tts', 'mms'], default='f5tts',
                        help='Type of model to use: f5tts (F5-TTS Romanian) or mms (facebook/mms-tts-ron) (default: f5tts)')
    
    args = parser.parse_args()
    
    print(f"{'='*60}")
    print(f"Full Dataset Audio Generation")
    print(f"{'='*60}")
    print(f"Model: {args.model_type.upper()}")
    
    # Check single reference mode
    if args.single_ref_audio and args.single_ref_text:
        print(f"Using single reference audio: {args.single_ref_audio}")
        print(f"Reference text: {args.single_ref_text}")
        if args.model_type == "f5tts":
            print("Mode: Voice conversion/style transfer")
        else:
            print("Mode: Text-to-speech generation (MMS-TTS-RON doesn't use reference audio)")
    else:
        if args.model_type == "f5tts":
            print("Mode: Self-reconstruction (each sample uses its own audio as reference)")
        else:
            print("Mode: Text-to-speech generation (MMS-TTS-RON doesn't use reference audio)")
    
    # Load dataset metadata
    df = load_metadata(args.metadata_path)
    
    # Apply sample limitations if specified
    if args.max_samples:
        end_idx = min(args.start_from + args.max_samples, len(df))
        df = df.iloc[args.start_from:end_idx]
        print(f"Processing samples {args.start_from} to {end_idx-1} ({len(df)} samples)")
    else:
        df = df.iloc[args.start_from:]
        print(f"Processing all samples from index {args.start_from} ({len(df)} samples)")
    
    # Create output structure
    output_dir, generated_dir, reference_dir, metadata_dir = create_output_structure(args.output_dir)
    
    # Load model based on type
    if args.model_type == "mms":
        model, vocoder = load_mms_romanian_model()  # vocoder is tokenizer for MMS
    else:
        model, vocoder = load_romanian_model()
    
    # Process in batches
    all_generated_files = []
    all_reference_files = []
    
    total_batches = (len(df) + args.batch_size - 1) // args.batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, len(df))
        df_batch = df.iloc[start_idx:end_idx]
        
        generated_files, reference_files = generate_audio_batch(
            model, vocoder, df_batch, generated_dir, reference_dir, batch_idx, total_batches,
            args.single_ref_audio, args.single_ref_text, args.model_type
        )
        
        all_generated_files.extend(generated_files)
        all_reference_files.extend(reference_files)
        
        print(f"Batch {batch_idx + 1} completed: {len(generated_files)} successful generations")
    
    # Save metadata
    metadata_file = save_generation_metadata(
        output_dir, metadata_dir, all_generated_files, all_reference_files, len(df), args.model_type
    )
    
    print(f"\n{'='*60}")
    print(f"Audio Generation Complete!")
    print(f"{'='*60}")
    print(f"Results saved to: {output_dir}")
    print(f"Total successful generations: {len(all_generated_files)}")
    print(f"Success rate: {len(all_generated_files)/len(df)*100:.2f}%")
    print(f"Metadata file: {metadata_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
