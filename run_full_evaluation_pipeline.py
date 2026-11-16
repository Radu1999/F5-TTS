"""
Full Evaluation Pipeline Runner

This script runs the complete pipeline:
1. Generate audio for the full ro_tts_common dataset
2. Evaluate the generated audio using Whisper and compute WER

Usage examples:
  # Run full pipeline with default settings
  python run_full_evaluation_pipeline.py

  # Run with custom settings
  python run_full_evaluation_pipeline.py --max_samples 1000 --batch_size 50

  # Run only generation
  python run_full_evaluation_pipeline.py --only_generation

  # Run only evaluation (requires existing generation output)
  python run_full_evaluation_pipeline.py --only_evaluation --generation_dir path/to/generation/output
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Command completed successfully!")
        if result.stdout:
            print("Output:", result.stdout)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error code {e.returncode}")
        if e.stdout:
            print("stdout:", e.stdout)
        if e.stderr:
            print("stderr:", e.stderr)
        return False, e.stderr


def find_latest_generation_dir(base_dir="full_dataset_generation"):
    """Find the most recent generation directory"""
    base_path = Path(base_dir)
    if not base_path.exists():
        return None
    
    # Look for directories matching the pattern
    pattern = "ro_tts_common_generated_*"
    matching_dirs = list(base_path.glob(pattern))
    
    if not matching_dirs:
        return None
    
    # Return the most recent one
    latest_dir = max(matching_dirs, key=lambda x: x.stat().st_mtime)
    return str(latest_dir)


def main():
    parser = argparse.ArgumentParser(description='Run full evaluation pipeline')
    
    # Pipeline control
    parser.add_argument('--only_generation', action='store_true',
                        help='Run only audio generation step (recommended first step)')
    parser.add_argument('--only_evaluation', action='store_true',
                        help='Run only evaluation step (requires existing generation)')
    parser.add_argument('--generate_and_store', action='store_true',
                        help='Generate and store all audio first, then exit (use --only_evaluation later)')
    
    # Generation parameters
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Batch size for processing (default: 100)')
    parser.add_argument('--max_samples', type=int, default=1000,
                        help='Maximum number of samples to process (default: all)')
    parser.add_argument('--start_from', type=int, default=0,
                        help='Start processing from this sample index (default: 0)')
    parser.add_argument('--generation_output_dir', type=str, default="full_dataset_generation",
                        help='Base output directory for generation (default: full_dataset_generation)')
    
    # Evaluation parameters
    parser.add_argument('--generation_dir', type=str, default=None,
                        help='Directory containing generated audio (auto-detect if not specified)')
    parser.add_argument('--eval_batch_size', type=int, default=50,
                        help='Batch size for evaluation (default: 50)')
    parser.add_argument('--whisper_model', type=str, default="gigant/whisper-medium-romanian",
                        help='Whisper model for transcription')
    parser.add_argument('--evaluation_output_dir', type=str, default=None,
                        help='Output directory for evaluation results (default: same as generation_dir)')
    parser.add_argument('--skip_generated_eval', action='store_true',
                        help='Skip evaluation of generated audio (only evaluate reference)')
    parser.add_argument('--skip_reference_eval', action='store_true',
                        help='Skip evaluation of reference audio (only evaluate generated)')
    
    # Data parameters
    parser.add_argument('--metadata_path', type=str, default="data/ro_tts_common_pinyin/metadata.csv",
                        help='Path to metadata CSV file')
    
    args = parser.parse_args()
    
    # Validate arguments
    exclusive_args = sum([args.only_generation, args.only_evaluation, args.generate_and_store])
    if exclusive_args > 1:
        print("Error: Can only specify one of --only_generation, --only_evaluation, or --generate_and_store")
        sys.exit(1)
    
    # Set flags based on arguments
    if args.generate_and_store:
        args.only_generation = True  # Generate and store implies only generation
    
    print(f"{'='*80}")
    print(f"Full Evaluation Pipeline")
    print(f"{'='*80}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    generation_dir = None
    
    # Step 1: Audio Generation
    # if not args.only_evaluation:
    #     print(f"\nStep 1: Audio Generation")
    #     print(f"{'='*40}")
    #
    #     gen_cmd = [
    #         sys.executable, "generate_full_dataset_audio.py",
    #         "--batch_size", str(args.batch_size),
    #         "--start_from", str(args.start_from),
    #         "--output_dir", args.generation_output_dir,
    #         "--metadata_path", args.metadata_path
    #     ]
    #
    #     if args.max_samples:
    #         gen_cmd.extend(["--max_samples", str(args.max_samples)])
    #
    #     success, output = run_command(gen_cmd, "Audio Generation")
    #
    #     if not success:
    #         print("Audio generation failed. Aborting pipeline.")
    #         sys.exit(1)
    #
    #     # Find the generated output directory
    #     generation_dir = find_latest_generation_dir(args.generation_output_dir)
    #     if generation_dir:
    #         print(f"Generated audio saved to: {generation_dir}")
    #         if args.generate_and_store:
    #             print(f"\n{'='*80}")
    #             print(f"GENERATION COMPLETE - AUDIO STORED")
    #             print(f"{'='*80}")
    #             print(f"All audio has been generated and stored.")
    #             print(f"To evaluate this generation, run:")
    #             print(f"")
    #             print(f"python run_full_evaluation_pipeline.py --only_evaluation \\")
    #             print(f"    --generation_dir \"{generation_dir}\"")
    #             print(f"")
    #             print(f"Or evaluate with custom settings:")
    #             print(f"python evaluate_wer_with_whisper.py \"{generation_dir}\" --batch_size 25")
    #             print(f"{'='*80}")
    #             return  # Exit after generation when using --generate_and_store
    #     else:
    #         print("Warning: Could not locate generation output directory")
    #
    # Step 2: WER Evaluation
    if not args.only_generation:
        print(f"\nStep 2: WER Evaluation")
        print(f"{'='*40}")
        
        # Determine generation directory
        if args.generation_dir:
            generation_dir = args.generation_dir
        elif generation_dir is None:
            generation_dir = find_latest_generation_dir(args.generation_output_dir)
        
        if not generation_dir or not Path(generation_dir).exists():
            print("Error: Could not find generation directory for evaluation")
            print("Please specify --generation_dir or run generation first")
            sys.exit(1)
        
        eval_cmd = [
            sys.executable, "evaluate_wer_with_whisper.py",
            "--generation_dir", generation_dir,
            "--batch_size", str(args.eval_batch_size),
            "--whisper_model", args.whisper_model
        ]
        
        if args.max_samples:
            eval_cmd.extend(["--max_samples", str(args.max_samples)])
        
        if args.evaluation_output_dir:
            eval_cmd.extend(["--output_dir", args.evaluation_output_dir])
        
        if args.skip_generated_eval:
            eval_cmd.append("--skip_generated")
        
        if args.skip_reference_eval:
            eval_cmd.append("--skip_reference")
        
        success, output = run_command(eval_cmd, "WER Evaluation")
        
        if not success:
            print("WER evaluation failed.")
            sys.exit(1)
    
    # Summary
    print(f"\n{'='*80}")
    print(f"Pipeline Complete!")
    print(f"{'='*80}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if generation_dir:
        print(f"Generation output: {generation_dir}")
        
        # Look for evaluation results
        eval_dirs = list(Path(generation_dir).glob("wer_evaluation_*"))
        if eval_dirs:
            latest_eval = max(eval_dirs, key=lambda x: x.stat().st_mtime)
            print(f"Evaluation output: {latest_eval}")
            
            # Try to show summary if available
            summary_file = latest_eval / "wer_evaluation_summary.txt"
            if summary_file.exists():
                print(f"\nEvaluation Summary:")
                print("-" * 40)
                with open(summary_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    # Show key metrics
                    for line in lines:
                        if "Word Error Rate" in line or "Match Error Rate" in line:
                            print(line.strip())
    
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
