#!/usr/bin/env python3
"""
Example Workflow Script for F5-TTS Romanian Evaluation

This script demonstrates the recommended two-step process:
1. Generate and store audio
2. Evaluate stored audio

Usage:
    python example_workflow.py --step 1  # Generate audio
    python example_workflow.py --step 2  # Evaluate audio
    python example_workflow.py --step all  # Run both steps
"""

import subprocess
import sys
import argparse
from pathlib import Path
import time


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True)
        elapsed = time.time() - start_time
        print(f"\n✅ {description} completed successfully in {elapsed/60:.1f} minutes")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n❌ {description} failed after {elapsed/60:.1f} minutes (exit code {e.returncode})")
        return False


def find_latest_generation_dir(base_dir="full_dataset_generation"):
    """Find the most recent generation directory"""
    base_path = Path(base_dir)
    if not base_path.exists():
        return None
    
    pattern = "ro_tts_common_generated_*"
    matching_dirs = list(base_path.glob(pattern))
    
    if not matching_dirs:
        return None
    
    latest_dir = max(matching_dirs, key=lambda x: x.stat().st_mtime)
    return str(latest_dir)


def step1_generate_audio(args):
    """Step 1: Generate and store audio"""
    print(f"\n🎵 STEP 1: GENERATE AND STORE AUDIO")
    print(f"{'='*80}")
    
    cmd = [
        sys.executable, "run_full_evaluation_pipeline.py",
        "--generate_and_store",
        "--batch_size", str(args.batch_size),
        "--max_samples", str(args.max_samples) if args.max_samples else "0"
    ]
    
    if args.max_samples:
        cmd[5] = str(args.max_samples)
    else:
        cmd = cmd[:5] + cmd[6:]  # Remove max_samples if not specified
    
    success = run_command(cmd, "Audio Generation")
    
    if success:
        generation_dir = find_latest_generation_dir()
        if generation_dir:
            print(f"\n📁 Generated audio stored in: {generation_dir}")
            print(f"💾 Storage used: ~{args.max_samples * 2 if args.max_samples else 14485 * 2}MB")
            return generation_dir
        else:
            print("\n⚠️  Warning: Could not locate generation directory")
            return None
    else:
        print("\n❌ Audio generation failed")
        return None


def step2_evaluate_audio(args, generation_dir=None):
    """Step 2: Evaluate stored audio"""
    print(f"\n📊 STEP 2: EVALUATE STORED AUDIO")
    print(f"{'='*80}")
    
    if not generation_dir:
        generation_dir = find_latest_generation_dir()
        if not generation_dir:
            print("❌ No generation directory found. Please run Step 1 first.")
            return False
    
    print(f"📂 Evaluating audio from: {generation_dir}")
    
    cmd = [
        sys.executable, "run_full_evaluation_pipeline.py",
        "--only_evaluation",
        "--generation_dir", generation_dir,
        "--eval_batch_size", str(args.eval_batch_size)
    ]
    
    if args.max_samples:
        cmd.extend(["--max_samples", str(args.max_samples)])
    
    success = run_command(cmd, "Audio Evaluation")
    
    if success:
        print(f"\n✅ Evaluation completed successfully")
        print(f"📈 Check the results in the evaluation directory within: {generation_dir}")
        return True
    else:
        print(f"\n❌ Evaluation failed")
        return False


def main():
    parser = argparse.ArgumentParser(description='Example workflow for F5-TTS evaluation')
    parser.add_argument('--step', choices=['1', '2', 'all'], default='all',
                        help='Which step to run: 1=generate, 2=evaluate, all=both (default: all)')
    parser.add_argument('--max_samples', type=int, default=100,
                        help='Maximum number of samples to process (default: 100, use 0 for all)')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size for generation (default: 50)')
    parser.add_argument('--eval_batch_size', type=int, default=25,
                        help='Batch size for evaluation (default: 25)')
    parser.add_argument('--generation_dir', type=str, default=None,
                        help='Generation directory for step 2 (auto-detect if not specified)')
    
    args = parser.parse_args()
    
    # Convert 0 to None for max_samples (means all samples)
    if args.max_samples == 0:
        args.max_samples = None
    
    print(f"🚀 F5-TTS Romanian Evaluation Workflow")
    print(f"{'='*80}")
    print(f"Step: {args.step}")
    print(f"Max samples: {args.max_samples or 'All (~14,485)'}")
    print(f"Generation batch size: {args.batch_size}")
    print(f"Evaluation batch size: {args.eval_batch_size}")
    
    start_time = time.time()
    generation_dir = None
    
    # Step 1: Generate audio
    if args.step in ['1', 'all']:
        generation_dir = step1_generate_audio(args)
        if not generation_dir and args.step == 'all':
            print("❌ Cannot proceed to evaluation due to generation failure")
            sys.exit(1)
    
    # Step 2: Evaluate audio
    if args.step in ['2', 'all']:
        if args.step == '2' and args.generation_dir:
            generation_dir = args.generation_dir
        
        success = step2_evaluate_audio(args, generation_dir)
        if not success:
            sys.exit(1)
    
    # Summary
    total_time = time.time() - start_time
    print(f"\n🎉 WORKFLOW COMPLETE!")
    print(f"{'='*80}")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
    
    if generation_dir:
        print(f"📁 Generation directory: {generation_dir}")
        
        # Show next steps
        if args.step == '1':
            print(f"\n📋 NEXT STEPS:")
            print(f"To evaluate this generated audio, run:")
            print(f"python example_workflow.py --step 2 --generation_dir \"{generation_dir}\"")
        elif args.step in ['2', 'all']:
            print(f"\n📋 ADDITIONAL EVALUATIONS:")
            print(f"You can run more evaluations on the same data:")
            print(f"python evaluate_wer_with_whisper.py \"{generation_dir}\" --skip_reference")
            print(f"python evaluate_wer_with_whisper.py \"{generation_dir}\" --skip_generated")


if __name__ == "__main__":
    main()
