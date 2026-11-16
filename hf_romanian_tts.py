import torch
import scipy.io.wavfile
from transformers import VitsModel, AutoTokenizer
from pathlib import Path
import pandas as pd
from datetime import datetime

def load_hf_romanian_model():
    """Load the HuggingFace Romanian TTS model"""
    print("Loading HuggingFace Romanian TTS model...")
    model = VitsModel.from_pretrained("facebook/mms-tts-ron")
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-ron")
    print("HuggingFace model loaded successfully!")
    return model, tokenizer

def generate_hf_tts(model, tokenizer, text, output_path):
    """Generate TTS audio using HuggingFace model"""
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        output = model(**inputs).waveform
    
    scipy.io.wavfile.write(
        output_path, 
        rate=model.config.sampling_rate, 
        data=output.squeeze().cpu().numpy()
    )
    
    return output_path, model.config.sampling_rate

def generate_hf_samples_for_speakers(speakers_csv="selected_speakers.csv", output_dir="hf_tts_outputs"):
    """Generate TTS samples using HuggingFace model for all speakers"""
    try:
        speakers_df = pd.read_csv(speakers_csv)
        print(f"Loaded {len(speakers_df)} speakers from {speakers_csv}")
    except FileNotFoundError:
        print(f"Error: Could not find {speakers_csv}")
        return None
    
    model, tokenizer = load_hf_romanian_model()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir) / f"hf_romanian_tts_{timestamp}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    generated_files = []
    
    print("Generating HuggingFace TTS samples...")
    for idx, row in speakers_df.iterrows():
        try:
            speaker_info = f"speaker_{idx+1}_{row['gender']}_{row['age']}"
            output_filename = f"{speaker_info}_hf_generated.wav"
            full_output_path = output_path / output_filename
            
            print(f"  Generating sample for speaker {idx+1}: {row['sentence'][:50]}...")
            
            audio_path, sample_rate = generate_hf_tts(
                model, tokenizer, row['sentence'], str(full_output_path)
            )
            
            generated_files.append({
                'speaker_id': idx + 1,
                'output_path': str(audio_path),
                'text': row['sentence'],
                'gender': row['gender'],
                'age': row['age'],
                'client_id': row['client_id'],
                'sample_rate': sample_rate,
                'model': 'facebook/mms-tts-ron'
            })
            
            print(f"    Saved: {output_filename}")
            
        except Exception as e:
            print(f"  Error generating sample for speaker {idx+1}: {e}")
            continue
    
    metadata_file = output_path / "hf_generation_metadata.csv"
    if generated_files:
        metadata_df = pd.DataFrame(generated_files)
        metadata_df.to_csv(metadata_file, index=False)
        print(f"Metadata saved to: {metadata_file}")
    
    print(f"\nGenerated {len(generated_files)} HuggingFace TTS samples")
    print(f"Results saved to: {output_path}")
    
    return output_path

if __name__ == "__main__":
    result_dir = generate_hf_samples_for_speakers()
    
    if result_dir:
        print(f"HuggingFace TTS generation completed successfully!")
        print(f"Check the results in: {result_dir}")
    else:
        print("HuggingFace TTS generation failed!")

