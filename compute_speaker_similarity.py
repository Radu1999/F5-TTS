"""
Speaker Similarity Computation Script

This script computes speaker similarity scores between reference and generated audio files
using the TitaNet-L model. It processes the output from generate_full_dataset_audio.py
and adds speaker similarity scores to the metadata.

Usage:
    python compute_speaker_similarity.py --input_dir full_dataset_generation/ro_tts_common_generated_TIMESTAMP
    python compute_speaker_similarity.py --input_dir full_dataset_generation/ro_tts_common_generated_TIMESTAMP --batch_size 50
"""

import pandas as pd
import json
import argparse
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np
from datetime import datetime
import logging

from speaker_similarity import create_speaker_similarity_scorer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_generation_metadata(input_dir):
    """Load the generation metadata from the specified directory"""
    metadata_dir = Path(input_dir) / "metadata"
    metadata_file = metadata_dir / "generation_metadata.json"
    
    if not metadata_file.exists():
        raise FileNotFoundError(f"Generation metadata file not found: {metadata_file}")
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    return metadata


def load_generated_samples_csv(input_dir):
    """Load the generated samples CSV file"""
    metadata_dir = Path(input_dir) / "metadata"
    csv_file = metadata_dir / "generated_samples.csv"
    
    if not csv_file.exists():
        raise FileNotFoundError(f"Generated samples CSV not found: {csv_file}")
    
    df = pd.read_csv(csv_file)
    return df


def validate_audio_files(df, input_dir):
    """Validate that all audio files exist"""
    missing_files = []
    
    for idx, row in df.iterrows():
        ref_path = './' + row['original_audio_path']
        gen_path = './' +  row['generated_audio_path']
        
        if not Path(ref_path).exists():
            missing_files.append(f"Reference: {ref_path}")
        if not Path(gen_path).exists():
            missing_files.append(f"Generated: {gen_path}")
    
    if missing_files:
        logger.warning(f"Found {len(missing_files)} missing audio files:")
        for file in missing_files[:10]:  # Show first 10
            logger.warning(f"  {file}")
        if len(missing_files) > 10:
            logger.warning(f"  ... and {len(missing_files) - 10} more")
    
    return len(missing_files) == 0


def compute_similarity_scores(df, input_dir, speaker_scorer, batch_size=32):
    """Compute speaker similarity scores for all samples"""
    logger.info(f"Computing speaker similarity scores for {len(df)} samples...")
    
    similarity_scores = []
    failed_computations = []
    
    # Process in batches for better progress tracking
    total_batches = (len(df) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(df))
        batch_df = df.iloc[start_idx:end_idx]
        
        logger.info(f"Processing batch {batch_idx + 1}/{total_batches} (samples {start_idx}-{end_idx-1})")
        
        batch_scores = []
        
        for idx, row in tqdm(batch_df.iterrows(), total=len(batch_df), 
                           desc=f"Batch {batch_idx + 1}/{total_batches}"):
            try:
                ref_path = './' + row['original_audio_path']
                gen_path = './' + row['generated_audio_path']

                
                # Check if files exist
                if not Path(ref_path).exists() or not Path(gen_path).exists():
                    logger.warning(f"Missing audio files for sample {idx}, skipping")
                    batch_scores.append(None)
                    failed_computations.append(idx)
                    continue
                
                # Compute similarity
                similarity = speaker_scorer.compute_similarity(str(ref_path), str(gen_path))
                batch_scores.append(similarity)
                
            except Exception as e:
                logger.error(f"Failed to compute similarity for sample {idx}: {e}")
                batch_scores.append(None)
                failed_computations.append(idx)
        
        similarity_scores.extend(batch_scores)
        logger.info(f"Batch {batch_idx + 1} completed: {sum(1 for s in batch_scores if s is not None)}/{len(batch_scores)} successful")
    
    logger.info(f"Similarity computation completed:")
    logger.info(f"  Total samples: {len(df)}")
    logger.info(f"  Successful computations: {sum(1 for s in similarity_scores if s is not None)}")
    logger.info(f"  Failed computations: {len(failed_computations)}")
    
    return similarity_scores, failed_computations


def compute_similarity_statistics(similarity_scores):
    """Compute statistics for the similarity scores"""
    valid_scores = [s for s in similarity_scores if s is not None]
    
    if not valid_scores:
        return None
    
    stats = {
        'mean_similarity': float(np.mean(valid_scores)),
        'std_similarity': float(np.std(valid_scores)),
        'min_similarity': float(np.min(valid_scores)),
        'max_similarity': float(np.max(valid_scores)),
        'median_similarity': float(np.median(valid_scores)),
        'num_computed': len(valid_scores),
        'num_failed': len(similarity_scores) - len(valid_scores),
        'success_rate': len(valid_scores) / len(similarity_scores)
    }
    
    return stats


def save_results(input_dir, df, similarity_scores, metadata, stats):
    """Save the results with speaker similarity scores"""
    output_dir = Path(input_dir)
    metadata_dir = output_dir / "metadata"
    
    # Add similarity scores to the DataFrame
    df_with_similarity = df.copy()
    df_with_similarity['speaker_similarity_score'] = similarity_scores
    
    # Save updated CSV
    csv_output_path = metadata_dir / "generated_samples_with_similarity.csv"
    df_with_similarity.to_csv(csv_output_path, index=False)
    logger.info(f"Saved updated CSV with similarity scores: {csv_output_path}")
    
    # Update metadata
    updated_metadata = metadata.copy()
    updated_metadata['generation_info']['speaker_similarity_enabled'] = True
    updated_metadata['generation_info']['speaker_similarity_stats'] = stats
    updated_metadata['generation_info']['similarity_computation_timestamp'] = datetime.now().isoformat()
    
    # Update generated samples in metadata
    for i, sample in enumerate(updated_metadata['generated_samples']):
        if i < len(similarity_scores):
            sample['speaker_similarity_score'] = similarity_scores[i]
    
    # Save updated metadata
    metadata_output_path = metadata_dir / "generation_metadata_with_similarity.json"
    with open(metadata_output_path, 'w', encoding='utf-8') as f:
        json.dump(updated_metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved updated metadata with similarity scores: {metadata_output_path}")
    
    # Save summary
    summary_output_path = metadata_dir / "speaker_similarity_summary.txt"
    with open(summary_output_path, 'w', encoding='utf-8') as f:
        f.write(f"Speaker Similarity Analysis Summary\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Input Directory: {input_dir}\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model Used: TitaNet-L\n\n")
        
        if stats:
            f.write(f"Statistics:\n")
            f.write(f"  Total Samples: {stats['num_computed'] + stats['num_failed']}\n")
            f.write(f"  Successfully Computed: {stats['num_computed']}\n")
            f.write(f"  Failed Computations: {stats['num_failed']}\n")
            f.write(f"  Success Rate: {stats['success_rate']*100:.2f}%\n\n")
            f.write(f"Similarity Score Statistics:\n")
            f.write(f"  Mean: {stats['mean_similarity']:.4f} ({stats['mean_similarity']*100:.2f}%)\n")
            f.write(f"  Standard Deviation: {stats['std_similarity']:.4f}\n")
            f.write(f"  Minimum: {stats['min_similarity']:.4f} ({stats['min_similarity']*100:.2f}%)\n")
            f.write(f"  Maximum: {stats['max_similarity']:.4f} ({stats['max_similarity']*100:.2f}%)\n")
            f.write(f"  Median: {stats['median_similarity']:.4f} ({stats['median_similarity']*100:.2f}%)\n\n")
        else:
            f.write(f"No valid similarity scores computed.\n\n")
        
        f.write(f"Output Files:\n")
        f.write(f"  Updated CSV: {csv_output_path.name}\n")
        f.write(f"  Updated Metadata: {metadata_output_path.name}\n")
        f.write(f"  This Summary: {summary_output_path.name}\n")
    
    logger.info(f"Saved analysis summary: {summary_output_path}")
    
    return csv_output_path, metadata_output_path, summary_output_path


def main():
    parser = argparse.ArgumentParser(description='Compute speaker similarity scores for generated audio')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory containing generated audio and metadata (e.g., full_dataset_generation/ro_tts_common_generated_TIMESTAMP)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for processing samples (default: 32)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use for TitaNet-L model (cpu, cuda, or auto) (default: auto)')
    parser.add_argument('--validate_files', action='store_true',
                        help='Validate that all audio files exist before processing')
    
    args = parser.parse_args()
    
    # Validate input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    logger.info(f"{'='*60}")
    logger.info(f"Speaker Similarity Computation")
    logger.info(f"{'='*60}")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Device: {args.device}")
    
    try:
        # Load metadata and samples
        logger.info("Loading generation metadata...")
        metadata = load_generation_metadata(input_dir)
        
        logger.info("Loading generated samples...")
        df = load_generated_samples_csv(input_dir)
        logger.info(f"Loaded {len(df)} generated samples")
        
        # Validate audio files if requested
        if args.validate_files:
            logger.info("Validating audio files...")
            files_valid = validate_audio_files(df, input_dir)
            if not files_valid:
                logger.warning("Some audio files are missing. Continuing anyway...")
        
        # Initialize speaker similarity scorer
        logger.info("Initializing TitaNet-L speaker similarity scorer...")
        speaker_scorer = create_speaker_similarity_scorer(device=args.device)
        if speaker_scorer is None:
            logger.error("Failed to create speaker similarity scorer. Make sure NVIDIA NeMo is installed.")
            sys.exit(1)
        
        speaker_scorer.load_model()
        logger.info("TitaNet-L model loaded successfully!")
        
        # Compute similarity scores
        similarity_scores, failed_computations = compute_similarity_scores(
            df, input_dir, speaker_scorer, args.batch_size
        )
        
        # Compute statistics
        stats = compute_similarity_statistics(similarity_scores)
        
        # Save results
        logger.info("Saving results...")
        csv_path, metadata_path, summary_path = save_results(
            input_dir, df, similarity_scores, metadata, stats
        )
        
        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info(f"Speaker Similarity Computation Complete!")
        logger.info(f"{'='*60}")
        
        if stats:
            logger.info(f"Results Summary:")
            logger.info(f"  Total samples: {stats['num_computed'] + stats['num_failed']}")
            logger.info(f"  Successfully computed: {stats['num_computed']}")
            logger.info(f"  Failed computations: {stats['num_failed']}")
            logger.info(f"  Success rate: {stats['success_rate']*100:.2f}%")
            logger.info(f"  Mean similarity: {stats['mean_similarity']:.4f} ({stats['mean_similarity']*100:.2f}%)")
            logger.info(f"  Std deviation: {stats['std_similarity']:.4f}")
            logger.info(f"  Min similarity: {stats['min_similarity']:.4f} ({stats['min_similarity']*100:.2f}%)")
            logger.info(f"  Max similarity: {stats['max_similarity']:.4f} ({stats['max_similarity']*100:.2f}%)")
        else:
            logger.warning("No valid similarity scores were computed.")
        
        logger.info(f"\nOutput files:")
        logger.info(f"  CSV with similarity scores: {csv_path}")
        logger.info(f"  Updated metadata: {metadata_path}")
        logger.info(f"  Analysis summary: {summary_path}")
        logger.info(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
