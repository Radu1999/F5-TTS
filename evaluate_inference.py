import pandas as pd
import os
import torch
import soundfile as sf
import torchaudio
import shutil
from datetime import datetime
from pathlib import Path
import json
import scipy.io.wavfile
from transformers import VitsModel, AutoTokenizer

from f5_tts.infer.utils_infer import (
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text
)
from f5_tts.model import DiT

TARGET_SAMPLE_RATE = 24000


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

    ckpt_path = "ckpts/ro_tts_lang_mod/pretrained_model_1250000.safetensors"
    vocab_file = "data/ro_tts_pinyin/vocab.txt"

    model = load_model(
        model_cls=DiT,
        model_cfg=model_cfg,
        use_ema=True,
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


def load_hf_romanian_model():
    """Load the HuggingFace Romanian TTS model"""
    print("Loading HuggingFace Romanian TTS model...")
    model = VitsModel.from_pretrained("facebook/mms-tts-ron")
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-ron")
    print("HuggingFace model loaded successfully!")
    return model, tokenizer


def generate_hf_audio_samples(model, tokenizer, speakers_df, generated_dir):
    """Generate audio samples using HuggingFace model"""
    print("Generating HuggingFace TTS audio samples...")
    generated_files = []
    
    for idx, row in speakers_df.iterrows():
        try:
            print(f"  Generating HF sample for speaker {idx+1}: {row['generate_sentence'][:50]}...")
            
            inputs = tokenizer(row['generate_sentence'], return_tensors="pt")
            
            with torch.no_grad():
                output = model(**inputs).waveform
            
            speaker_info = f"speaker_{idx+1}_{row['gender']}_{row['age']}"
            output_filename = f"{speaker_info}_hf_generated.wav"
            output_path = generated_dir / output_filename
            
            scipy.io.wavfile.write(
                str(output_path), 
                rate=model.config.sampling_rate, 
                data=output.squeeze().cpu().numpy()
            )
            
            generated_files.append({
                'speaker_id': idx + 1,
                'output_path': str(output_path),
                'reference_text': row['sentence'],
                'generated_text': row['generate_sentence'],
                'gender': row['gender'],
                'age': row['age'],
                'client_id': row['client_id'],
                'sample_rate': model.config.sampling_rate,
                'model': 'facebook/mms-tts-ron'
            })
            
            print(f"    Saved: {output_filename}")
            
        except Exception as e:
            print(f"  Error generating HF sample for speaker {idx+1}: {e}")
            continue
    
    return generated_files


def load_selected_speakers(csv_path="selected_speakers.csv"):
    """Load selected speakers from CSV file"""
    try:
        df = pd.read_csv(csv_path, sep=';')
        print(f"Loaded {len(df)} selected speakers from {csv_path}")
        return df
    except FileNotFoundError:
        print(f"Error: Could not find {csv_path}")
        return None
    except Exception as e:
        print(f"Error loading speakers: {e}")
        return None


def create_evaluation_folder_structure(model_name, base_output_dir="evaluation_outputs"):
    """Create organized folder structure for model evaluation"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = Path(base_output_dir) / f"{model_name}_{timestamp}"
    
    generated_dir = model_dir / "generated_samples"
    ground_truth_dir = model_dir / "ground_truth"
    metadata_dir = model_dir / "metadata"
    
    for dir_path in [generated_dir, ground_truth_dir, metadata_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Created evaluation directory structure at: {model_dir}")
    return model_dir, generated_dir, ground_truth_dir, metadata_dir


def get_audio_file_path(csv_filename, wavs_dir="data/ro_tts_common_pinyin/wavs"):
    """Convert CSV filename from .mp3 to .wav format and get full path"""
    wav_filename = csv_filename.replace('.mp3', '.wav')
    return Path(wavs_dir) / wav_filename


def copy_ground_truth_files(speakers_df, ground_truth_dir, wavs_dir="data/ro_tts_common_pinyin/wavs"):
    """Process and copy original audio files as ground truth references"""
    print("Processing ground truth audio files...")
    copied_files = []
    
    for idx, row in speakers_df.iterrows():
        source_path = get_audio_file_path(row['path'], wavs_dir)
        
        if source_path.exists():
            try:
                processed_audio_path, processed_text = preprocess_ref_audio_text(
                    str(source_path), 
                    row['sentence']
                )
                
                speaker_info = f"speaker_{idx+1}_{row['gender']}_{row['age']}"
                dest_filename = f"{speaker_info}_ground_truth.wav"
                dest_path = ground_truth_dir / dest_filename
                
                shutil.copy2(processed_audio_path, str(dest_path))
                
                copied_files.append({
                    'speaker_id': idx + 1,
                    'original_path': str(source_path),
                    'copied_path': str(dest_path),
                    'reference_text': row['sentence'],
                    'generated_text': row['generate_sentence'],
                    'processed_text': processed_text,
                    'gender': row['gender'],
                    'age': row['age'],
                    'client_id': row['client_id'],
                    'processed_sample_rate': TARGET_SAMPLE_RATE,
                    'preprocessing_method': 'preprocess_ref_audio_text'
                })
                print(f"  Processed: {source_path.name} -> {dest_filename}")
                
            except Exception as e:
                print(f"  Error processing {source_path.name}: {e}")
                continue
        else:
            print(f"  Warning: Audio file not found: {source_path}")
    
    return copied_files


def generate_audio_samples(model, vocoder, speakers_df, generated_dir, wavs_dir="data/ro_tts_common_pinyin/wavs"):
    """Generate audio samples for each speaker using the model"""
    print("Generating audio samples...")
    generated_files = []
    device = next(model.parameters()).device
    
    for idx, row in speakers_df.iterrows():
        try:
            ref_audio_path = get_audio_file_path(row['path'], wavs_dir)
            
            if not ref_audio_path.exists():
                print(f"  Warning: Reference audio not found for speaker {idx+1}: {ref_audio_path}")
                continue
            
            # Use 'sentence' column as reference text for the audio
            ref_audio, ref_text = preprocess_ref_audio_text(
                str(ref_audio_path), 
                row['sentence']
            )
            
            audio, sample_rate = torchaudio.load(str(ref_audio_path))
            print(f"  Generating sample for speaker {idx+1}: {row['generate_sentence'][:50]}...")
            
            # Generate audio for 'generate_sentence' column using 'sentence' as reference
            generated_audio, sample_rate, _ = infer_process(
                ref_audio=ref_audio,
                ref_text=ref_text,
                gen_text=row['generate_sentence'],
                model_obj=model,
                vocoder=vocoder,
                speed=0.95,
                device=device,
                show_info=print,
                cross_fade_duration=0.15,
                nfe_step=32,
                cfg_strength=2.0
            )
            
            speaker_info = f"speaker_{idx+1}_{row['gender']}_{row['age']}"
            output_filename = f"{speaker_info}_generated.wav"
            output_path = generated_dir / output_filename
            
            sf.write(output_path, generated_audio, sample_rate)
            
            generated_files.append({
                'speaker_id': idx + 1,
                'output_path': str(output_path),
                'reference_path': str(ref_audio_path),
                'reference_text': row['sentence'],
                'generated_text': row['generate_sentence'],
                'gender': row['gender'],
                'age': row['age'],
                'client_id': row['client_id']
            })
            
            print(f"    Saved: {output_filename}")
            
        except Exception as e:
            print(f"  Error generating sample for speaker {idx+1}: {e}")
            continue
    
    return generated_files


def save_evaluation_metadata(model_dir, metadata_dir, model_name, generated_files, ground_truth_files):
    """Save evaluation metadata and information"""
    metadata = {
        'evaluation_info': {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'total_speakers': len(generated_files),
            'successful_generations': len(generated_files),
            'ground_truth_files': len(ground_truth_files),
            'target_sample_rate': TARGET_SAMPLE_RATE
        },
        'generated_samples': generated_files,
        'ground_truth_samples': ground_truth_files
    }
    
    metadata_file = metadata_dir / "evaluation_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    if generated_files:
        generated_df = pd.DataFrame(generated_files)
        generated_df.to_csv(metadata_dir / "generated_samples.csv", index=False)
    
    if ground_truth_files:
        ground_truth_df = pd.DataFrame(ground_truth_files)
        ground_truth_df.to_csv(metadata_dir / "ground_truth_samples.csv", index=False)
    
    summary_file = metadata_dir / "evaluation_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"Model Evaluation Summary\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Output Directory: {model_dir}\n\n")
        f.write(f"Statistics:\n")
        f.write(f"  Total Speakers: {len(generated_files)}\n")
        f.write(f"  Successful Generations: {len(generated_files)}\n")
        f.write(f"  Ground Truth Files: {len(ground_truth_files)}\n")
        f.write(f"  Target Sample Rate: {TARGET_SAMPLE_RATE}Hz\n\n")
        
        f.write(f"Generation Method:\n")
        f.write(f"  Reference Text: 'sentence' column (for voice cloning)\n")
        f.write(f"  Generated Text: 'generate_sentence' column (target speech)\n\n")
        
        if generated_files:
            genders = [item['gender'] for item in generated_files]
            ages = [item['age'] for item in generated_files]
            f.write(f"Speaker Demographics:\n")
            f.write(f"  Genders: {pd.Series(genders).value_counts().to_dict()}\n")
            f.write(f"  Ages: {pd.Series(ages).value_counts().to_dict()}\n\n")
        
        f.write(f"Directory Structure:\n")
        f.write(f"  Generated Samples: {model_dir / 'generated_samples'}\n")
        f.write(f"  Ground Truth: {model_dir / 'ground_truth'}\n")
        f.write(f"  Metadata: {model_dir / 'metadata'}\n")
    
    print(f"Evaluation metadata saved to: {metadata_dir}")


def evaluate_model_with_selected_speakers(model_name="romanian_f5tts", speakers_csv="selected_speakers.csv"):
    """Complete evaluation pipeline for a model with selected speakers"""
    print(f"\n{'='*60}")
    print(f"Starting Model Evaluation: {model_name}")
    print(f"{'='*60}")
    
    speakers_df = load_selected_speakers(speakers_csv)
    if speakers_df is None:
        return None
    
    model_dir, generated_dir, ground_truth_dir, metadata_dir = create_evaluation_folder_structure(model_name)
    model, vocoder = load_romanian_model()
    ground_truth_files = copy_ground_truth_files(speakers_df, ground_truth_dir)
    generated_files = generate_audio_samples(model, vocoder, speakers_df, generated_dir)
    save_evaluation_metadata(model_dir, metadata_dir, model_name, generated_files, ground_truth_files)
    
    print(f"\n{'='*60}")
    print(f"Evaluation Complete!")
    print(f"Results saved to: {model_dir}")
    print(f"Generated {len(generated_files)} audio samples")
    print(f"Copied {len(ground_truth_files)} ground truth files")
    print(f"{'='*60}\n")
    
    return model_dir


def evaluate_hf_model_with_selected_speakers(model_name="romanian_hf_tts", speakers_csv="selected_speakers.csv"):
    """Complete evaluation pipeline using HuggingFace Romanian TTS model"""
    print(f"\n{'='*60}")
    print(f"Starting HuggingFace Model Evaluation: {model_name}")
    print(f"{'='*60}")
    
    speakers_df = load_selected_speakers(speakers_csv)
    if speakers_df is None:
        return None
    
    model_dir, generated_dir, ground_truth_dir, metadata_dir = create_evaluation_folder_structure(model_name)
    model, tokenizer = load_hf_romanian_model()
    ground_truth_files = copy_ground_truth_files(speakers_df, ground_truth_dir)
    generated_files = generate_hf_audio_samples(model, tokenizer, speakers_df, generated_dir)
    save_evaluation_metadata(model_dir, metadata_dir, model_name, generated_files, ground_truth_files)
    
    print(f"\n{'='*60}")
    print(f"HuggingFace Evaluation Complete!")
    print(f"Results saved to: {model_dir}")
    print(f"Generated {len(generated_files)} audio samples")
    print(f"Copied {len(ground_truth_files)} ground truth files")
    print(f"{'='*60}\n")
    
    return model_dir


if __name__ == "__main__":
    print("Choose TTS model for evaluation:")
    print("1. F5-TTS Romanian model (default)")
    print("2. HuggingFace facebook/mms-tts-ron model")
    print("3. Both models")
    
    choice = input("Enter your choice (1/2/3, default=1): ").strip() or "1"
    
    if choice == "1":
        result_dir = evaluate_model_with_selected_speakers(speakers_csv="selected_speakers_code_switch_noanno.csv")
        if result_dir:
            print(f"F5-TTS evaluation completed successfully!")
            print(f"Check the results in: {result_dir}")
        else:
            print("F5-TTS evaluation failed!")
    
    elif choice == "2":
        result_dir = evaluate_hf_model_with_selected_speakers(speakers_csv="selected_speakers_code_switch_noanno.csv")
        if result_dir:
            print(f"HuggingFace TTS evaluation completed successfully!")
            print(f"Check the results in: {result_dir}")
        else:
            print("HuggingFace TTS evaluation failed!")
    
    elif choice == "3":
        print("Running both evaluations...")
        f5_result = evaluate_model_with_selected_speakers()
        hf_result = evaluate_hf_model_with_selected_speakers()
        
        if f5_result and hf_result:
            print(f"Both evaluations completed successfully!")
            print(f"F5-TTS results: {f5_result}")
            print(f"HuggingFace results: {hf_result}")
        else:
            print("One or both evaluations failed!")
    
    else:
        print("Invalid choice. Running F5-TTS evaluation by default.")
        result_dir = evaluate_model_with_selected_speakers()
        if result_dir:
            print(f"F5-TTS evaluation completed successfully!")
            print(f"Check the results in: {result_dir}")
        else:
            print("F5-TTS evaluation failed!")

