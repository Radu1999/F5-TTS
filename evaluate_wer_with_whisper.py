"""
WER Evaluation Script using Whisper and jiwer

This script evaluates generated audio using Whisper for transcription and jiwer for WER computation.
It can work with the output from generate_full_dataset_audio.py or any similar audio generation structure.
"""

import pandas as pd
import os
import torch
import torchaudio
from datetime import datetime
from pathlib import Path
import json
import argparse
from tqdm import tqdm
import librosa
import numpy as np
import sys
import re

# Whisper imports
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# WER computation import
import jiwer

# Audio processing
import soundfile as sf

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


def load_whisper_model(model_name="gigant/whisper-medium-romanian"):
    """Load Whisper model for Romanian transcription"""
    print(f"Loading Whisper model: {model_name}")
    
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    
    # Set language and task
    model.config.decoder_ids = processor.get_decoder_prompt_ids(language="ro", task="transcribe")
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    print(f"Whisper model loaded successfully on {device}")
    return model, processor, device


def preprocess_audio_for_whisper(audio_path, target_sampling_rate=16000):
    """Load and preprocess audio for Whisper transcription"""
    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=target_sampling_rate)
        return audio
    except Exception as e:
        print(f"Error loading audio {audio_path}: {e}")
        return None


def transcribe_audio(audio, model, processor, device):
    """Transcribe audio using Whisper model"""
    try:
        # Prepare input features
        input_features = processor(
            audio, 
            return_tensors="pt", 
            sampling_rate=16000
        ).input_features
        
        # Move to device
        input_features = input_features.to(device)
        
        # Generate transcription
        with torch.no_grad():
            predicted_ids = model.generate(input_features, max_length=448)
        
        # Decode transcription
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        return transcription.strip()
    
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None


def normalize_text_for_wer(text):
    """Normalize text for WER computation"""
    if not text:
        return ""
    
    # First normalize Romanian characters
    text = normalize_romanian_text(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove common punctuation that doesn't affect pronunciation
    text = re.sub(r'[.,!?;:"„"«»”]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def compute_wer_metrics(reference_texts, hypothesis_texts):
    """Compute WER and related metrics using jiwer"""
    # Normalize texts
    ref_normalized = [normalize_text_for_wer(text) for text in reference_texts]
    hyp_normalized = [normalize_text_for_wer(text) for text in hypothesis_texts]
    
    # Remove empty pairs
    valid_pairs = [(ref, hyp) for ref, hyp in zip(ref_normalized, hyp_normalized) 
                   if ref and hyp]
    
    if not valid_pairs:
        return {
            'wer': 1.0,
            'mer': 1.0,
            'wil': 1.0,
            'wip': 0.0,
            'valid_samples': 0,
            'total_samples': len(reference_texts)
        }
    
    valid_refs, valid_hyps = zip(*valid_pairs)
    
    # Convert tuples to lists for jiwer
    valid_refs = list(valid_refs)
    valid_hyps = list(valid_hyps)
    
    # Compute metrics as average of individual sample metrics (correct approach)
    individual_wers = []
    individual_mers = []
    individual_wils = []
    individual_wips = []
    
    for ref, hyp in zip(valid_refs, valid_hyps):
        individual_wers.append(jiwer.wer(ref, hyp))
        individual_mers.append(jiwer.mer(ref, hyp))
        individual_wils.append(jiwer.wil(ref, hyp))
        individual_wips.append(jiwer.wip(ref, hyp))
    
    # Compute averages
    wer = sum(individual_wers) / len(individual_wers)
    mer = sum(individual_mers) / len(individual_mers)
    wil = sum(individual_wils) / len(individual_wils)
    wip = sum(individual_wips) / len(individual_wips)
    
    return {
        'wer': wer,
        'mer': mer,
        'wil': wil,
        'wip': wip,
        'valid_samples': len(valid_pairs),
        'total_samples': len(reference_texts)
    }


def load_generation_metadata(generation_dir):
    """Load metadata from audio generation output"""
    metadata_dir = Path(generation_dir) / "metadata"
    metadata_file = metadata_dir / "generation_metadata.json"
    
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Also load CSV for easier processing
    generated_csv = metadata_dir / "generated_samples.csv"
    if generated_csv.exists():
        df = pd.read_csv(generated_csv, encoding='utf-8')
    else:
        # Create DataFrame from JSON metadata
        df = pd.DataFrame(metadata['generated_samples'])
    
    # Normalize the text column if it exists
    if 'text' in df.columns:
        df['text'] = df['text'].apply(normalize_romanian_text)
    
    safe_print(f"Loaded metadata for {len(df)} generated samples")
    return metadata, df


def evaluate_wer_batch(df_batch, model, processor, device, batch_idx, total_batches, evaluate_reference=False):
    """Evaluate WER for a batch of audio samples"""
    results = []
    
    audio_type = "reference" if evaluate_reference else "generated"
    print(f"\nEvaluating {audio_type} batch {batch_idx + 1}/{total_batches} ({len(df_batch)} samples)")
    
    for idx, row in tqdm(df_batch.iterrows(), total=len(df_batch), desc=f"{audio_type.title()} Batch {batch_idx + 1}"):
        try:
            # Choose audio path based on evaluation type
            if evaluate_reference:
                audio_path = row.get('reference_audio_path') or row.get('original_audio_path')
                if not audio_path:
                    safe_print(f"  Warning: No reference audio path found for sample {row['sample_id']}")
                    continue
            else:
                audio_path = row['generated_audio_path']
            
            if not Path(audio_path).exists():
                safe_print(f"  Warning: {audio_type.title()} audio not found: {audio_path}")
                continue
            
            # Preprocess audio
            audio = preprocess_audio_for_whisper(audio_path)
            if audio is None:
                continue
            
            # Transcribe
            transcription = transcribe_audio(audio, model, processor, device)
            if transcription is None:
                continue
            
            # Normalize the transcription text
            transcription = normalize_romanian_text(transcription)
            
            # Compute WER for this single sample
            reference_text = row['text']
            wer_metrics = compute_wer_metrics([reference_text], [transcription])
            
            # Store results
            result_entry = {
                'sample_id': row['sample_id'],
                'reference_text': reference_text,
                'transcription': transcription,
                'reference_normalized': normalize_text_for_wer(reference_text),
                'transcription_normalized': normalize_text_for_wer(transcription),
                'wer': wer_metrics['wer'],
                'mer': wer_metrics['mer'],
                'wil': wer_metrics['wil'],
                'wip': wer_metrics['wip'],
                'client_id': row['client_id'],
                'audio_type': audio_type,
                'audio_path': audio_path
            }
            
            # Add the appropriate audio path field for backward compatibility
            if evaluate_reference:
                result_entry['reference_audio_path'] = audio_path
            else:
                result_entry['generated_audio_path'] = audio_path
                
            results.append(result_entry)
            
        except Exception as e:
            safe_print(f"  Error evaluating {audio_type} sample {row['sample_id']}: {e}")
            continue
    
    return results


def save_evaluation_results(output_dir, generated_results, reference_results, generation_metadata, generated_metrics, reference_metrics):
    """Save WER evaluation results for both generated and reference audio"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = Path(output_dir) / f"wer_evaluation_{timestamp}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results CSV for generated audio
    if generated_results:
        generated_df = pd.DataFrame(generated_results)
        generated_csv = eval_dir / "detailed_wer_results_generated.csv"
        generated_df.to_csv(generated_csv, index=False, encoding='utf-8')
        safe_print(f"Generated audio results saved to: {generated_csv}")
    
    # Save detailed results CSV for reference audio
    if reference_results:
        reference_df = pd.DataFrame(reference_results)
        reference_csv = eval_dir / "detailed_wer_results_reference.csv"
        reference_df.to_csv(reference_csv, index=False, encoding='utf-8')
        safe_print(f"Reference audio results saved to: {reference_csv}")
    
    # Combine all results for overall analysis
    all_results = generated_results + reference_results
    
    # Save summary metadata
    evaluation_metadata = {
        'evaluation_info': {
            'evaluation_date': datetime.now().isoformat(),
            'whisper_model': 'gigant/whisper-medium-romanian',
            'total_generated_samples': len(generated_results),
            'total_reference_samples': len(reference_results),
            'generation_metadata': generation_metadata['generation_info']
        },
        'generated_metrics': generated_metrics,
        'reference_metrics': reference_metrics,
        'generated_results': generated_results,
        'reference_results': reference_results
    }
    
    metadata_file = eval_dir / "evaluation_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_metadata, f, indent=2, ensure_ascii=False)
    
    # Save summary report
    summary_file = eval_dir / "wer_evaluation_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"WER Evaluation Summary\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Whisper Model: gigant/whisper-medium-romanian\n")
        f.write(f"Dataset: {generation_metadata['generation_info']['dataset']}\n")
        f.write(f"TTS Model: {generation_metadata['generation_info']['model_name']}\n\n")
        
        f.write(f"Sample Statistics:\n")
        f.write(f"  Generated Audio Samples: {len(generated_results)}\n")
        f.write(f"  Reference Audio Samples: {len(reference_results)}\n")
        f.write(f"  Total Samples Evaluated: {len(generated_results) + len(reference_results)}\n\n")
        
        # Generated Audio Metrics
        if generated_results:
            f.write(f"GENERATED AUDIO METRICS (TTS Quality):\n")
            f.write(f"{'='*50}\n")
            f.write(f"  Word Error Rate (WER): {generated_metrics['wer']:.4f} ({generated_metrics['wer']*100:.2f}%)\n")
            f.write(f"  Match Error Rate (MER): {generated_metrics['mer']:.4f} ({generated_metrics['mer']*100:.2f}%)\n")
            f.write(f"  Word Information Lost (WIL): {generated_metrics['wil']:.4f} ({generated_metrics['wil']*100:.2f}%)\n")
            f.write(f"  Word Information Preserved (WIP): {generated_metrics['wip']:.4f} ({generated_metrics['wip']*100:.2f}%)\n")
            f.write(f"  Valid Samples: {generated_metrics['valid_samples']}/{generated_metrics['total_samples']}\n\n")
            
            gen_wer_values = [r['wer'] for r in generated_results if r['wer'] is not None]
            if gen_wer_values:
                f.write(f"  Generated WER Distribution:\n")
                f.write(f"    Mean: {np.mean(gen_wer_values):.4f}\n")
                f.write(f"    Median: {np.median(gen_wer_values):.4f}\n")
                f.write(f"    Std: {np.std(gen_wer_values):.4f}\n")
                f.write(f"    Min: {np.min(gen_wer_values):.4f}\n")
                f.write(f"    Max: {np.max(gen_wer_values):.4f}\n\n")
        
        # Reference Audio Metrics (Whisper Baseline)
        if reference_results:
            f.write(f"REFERENCE AUDIO METRICS (Whisper Baseline):\n")
            f.write(f"{'='*50}\n")
            f.write(f"  Word Error Rate (WER): {reference_metrics['wer']:.4f} ({reference_metrics['wer']*100:.2f}%)\n")
            f.write(f"  Match Error Rate (MER): {reference_metrics['mer']:.4f} ({reference_metrics['mer']*100:.2f}%)\n")
            f.write(f"  Word Information Lost (WIL): {reference_metrics['wil']:.4f} ({reference_metrics['wil']*100:.2f}%)\n")
            f.write(f"  Word Information Preserved (WIP): {reference_metrics['wip']:.4f} ({reference_metrics['wip']*100:.2f}%)\n")
            f.write(f"  Valid Samples: {reference_metrics['valid_samples']}/{reference_metrics['total_samples']}\n\n")
            
            ref_wer_values = [r['wer'] for r in reference_results if r['wer'] is not None]
            if ref_wer_values:
                f.write(f"  Reference WER Distribution:\n")
                f.write(f"    Mean: {np.mean(ref_wer_values):.4f}\n")
                f.write(f"    Median: {np.median(ref_wer_values):.4f}\n")
                f.write(f"    Std: {np.std(ref_wer_values):.4f}\n")
                f.write(f"    Min: {np.min(ref_wer_values):.4f}\n")
                f.write(f"    Max: {np.max(ref_wer_values):.4f}\n\n")
        
        # Comparison if both are available
        if generated_results and reference_results:
            f.write(f"COMPARISON (Generated vs Reference):\n")
            f.write(f"{'='*50}\n")
            wer_diff = generated_metrics['wer'] - reference_metrics['wer']
            wer_diff_pct = (generated_metrics['wer'] / reference_metrics['wer'] - 1) * 100
            f.write(f"  WER Difference: {wer_diff:+.4f} ({wer_diff_pct:+.2f}%)\n")
            
            if wer_diff > 0:
                f.write(f"  → Generated audio has {wer_diff*100:.2f} percentage points higher WER\n")
            else:
                f.write(f"  → Generated audio has {abs(wer_diff)*100:.2f} percentage points lower WER\n")
            f.write(f"  → TTS Quality Factor: {reference_metrics['wer']/generated_metrics['wer']:.2f}x\n\n")
        
        f.write(f"Files Generated:\n")
        f.write(f"  Generated Results: {eval_dir / 'detailed_wer_results_generated.csv'}\n")
        f.write(f"  Reference Results: {eval_dir / 'detailed_wer_results_reference.csv'}\n")
        f.write(f"  Evaluation Metadata: {eval_dir / 'evaluation_metadata.json'}\n")
        f.write(f"  Summary Report: {eval_dir / 'wer_evaluation_summary.txt'}\n")
    
    print(f"\nEvaluation summary saved to: {summary_file}")
    return eval_dir


def main():
    parser = argparse.ArgumentParser(description='Evaluate WER using Whisper and jiwer')
    parser.add_argument('--generation_dir', type=str,
                        help='Directory containing generated audio and metadata')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Number of samples to process in each batch (default: 50)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to evaluate (default: all)')
    parser.add_argument('--whisper_model', type=str, default="gigant/whisper-medium-romanian",
                        help='Whisper model to use for transcription')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for evaluation results (default: same as generation_dir)')
    parser.add_argument('--skip_generated', action='store_true',
                        help='Skip evaluation of generated audio (only evaluate reference)')
    parser.add_argument('--skip_reference', action='store_true',
                        help='Skip evaluation of reference audio (only evaluate generated)')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.generation_dir
    
    if args.skip_generated and args.skip_reference:
        print("Error: Cannot skip both generated and reference audio evaluation")
        return
    
    print(f"{'='*70}")
    print(f"WER Evaluation with Whisper")
    print(f"{'='*70}")
    
    # Load generation metadata
    generation_metadata, df = load_generation_metadata(args.generation_dir)
    
    # Apply sample limitations if specified
    if args.max_samples:
        df = df.head(args.max_samples)
        safe_print(f"Evaluating first {len(df)} samples")
    else:
        safe_print(f"Evaluating all {len(df)} samples")
    
    # Load Whisper model
    model, processor, device = load_whisper_model(args.whisper_model)
    
    # Process in batches
    generated_results = []
    reference_results = []
    total_batches = (len(df) + args.batch_size - 1) // args.batch_size
    
    # Evaluate generated audio
    if not args.skip_generated:
        safe_print(f"\n{'='*50}")
        safe_print(f"Evaluating GENERATED audio (TTS Quality)")
        safe_print(f"{'='*50}")
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * args.batch_size
            end_idx = min(start_idx + args.batch_size, len(df))
            df_batch = df.iloc[start_idx:end_idx]
            
            batch_results = evaluate_wer_batch(
                df_batch, model, processor, device, batch_idx, total_batches, evaluate_reference=False
            )
            
            generated_results.extend(batch_results)
            print(f"Generated batch {batch_idx + 1} completed: {len(batch_results)} successful evaluations")
    
    # Evaluate reference audio (Whisper baseline)
    if not args.skip_reference:
        safe_print(f"\n{'='*50}")
        safe_print(f"Evaluating REFERENCE audio (Whisper Baseline)")
        safe_print(f"{'='*50}")
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * args.batch_size
            end_idx = min(start_idx + args.batch_size, len(df))
            df_batch = df.iloc[start_idx:end_idx]
            
            batch_results = evaluate_wer_batch(
                df_batch, model, processor, device, batch_idx, total_batches, evaluate_reference=True
            )
            
            reference_results.extend(batch_results)
            print(f"Reference batch {batch_idx + 1} completed: {len(batch_results)} successful evaluations")
    
    # Compute metrics for each type
    generated_metrics = {}
    reference_metrics = {}
    
    if generated_results:
        gen_reference_texts = [r['reference_text'] for r in generated_results]
        gen_transcriptions = [r['transcription'] for r in generated_results]
        generated_metrics = compute_wer_metrics(gen_reference_texts, gen_transcriptions)
    
    if reference_results:
        ref_reference_texts = [r['reference_text'] for r in reference_results]
        ref_transcriptions = [r['transcription'] for r in reference_results]
        reference_metrics = compute_wer_metrics(ref_reference_texts, ref_transcriptions)
    
    # Save results
    eval_dir = save_evaluation_results(
        args.output_dir, generated_results, reference_results, 
        generation_metadata, generated_metrics, reference_metrics
    )
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"WER Evaluation Complete!")
    print(f"{'='*70}")
    print(f"Results saved to: {eval_dir}")
    
    if generated_results:
        print(f"\nGENERATED AUDIO (TTS Quality):")
        print(f"  Successful evaluations: {len(generated_results)}")
        print(f"  WER: {generated_metrics['wer']:.4f} ({generated_metrics['wer']*100:.2f}%)")
        print(f"  MER: {generated_metrics['mer']:.4f} ({generated_metrics['mer']*100:.2f}%)")
    
    if reference_results:
        print(f"\nREFERENCE AUDIO (Whisper Baseline):")
        print(f"  Successful evaluations: {len(reference_results)}")
        print(f"  WER: {reference_metrics['wer']:.4f} ({reference_metrics['wer']*100:.2f}%)")
        print(f"  MER: {reference_metrics['mer']:.4f} ({reference_metrics['mer']*100:.2f}%)")
    
    if generated_results and reference_results:
        wer_diff = generated_metrics['wer'] - reference_metrics['wer']
        print(f"\nCOMPARISON:")
        print(f"  WER Difference: {wer_diff:+.4f} ({wer_diff*100:+.2f} percentage points)")
        if wer_diff > 0:
            print(f"  → Generated audio performs worse by {wer_diff*100:.2f} pp")
        else:
            print(f"  → Generated audio performs better by {abs(wer_diff)*100:.2f} pp")
    
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
