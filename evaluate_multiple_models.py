#!/usr/bin/env python3
"""
Multi-Model Evaluation Script

This script demonstrates how to evaluate multiple models with the same set of selected speakers.
Each model will get its own timestamped folder with organized outputs.

Usage:
    python evaluate_multiple_models.py
"""

from evaluate_inference import evaluate_model_with_selected_speakers
import os


def evaluate_multiple_models():
    """Evaluate multiple model configurations with selected speakers"""
    
    # Define different model configurations to test
    # You can modify these based on your available models
    model_configs = [
        {
            'name': 'romanian_f5tts_base',
            'description': 'Base Romanian F5-TTS model'
        },
        # Add more model configurations here as needed
        # {
        #     'name': 'romanian_f5tts_large', 
        #     'description': 'Large Romanian F5-TTS model'
        # },
        # {
        #     'name': 'romanian_f5tts_fine_tuned',
        #     'description': 'Fine-tuned Romanian F5-TTS model'
        # }
    ]
    
    print("Starting Multi-Model Evaluation")
    print("=" * 50)
    
    results = []
    
    for config in model_configs:
        print(f"\nEvaluating Model: {config['name']}")
        print(f"Description: {config['description']}")
        print("-" * 40)
        
        try:
            # Run evaluation for this model
            result_dir = evaluate_model_with_selected_speakers(
                model_name=config['name'],
                speakers_csv="selected_speakers.csv"
            )
            
            if result_dir:
                results.append({
                    'model_name': config['name'],
                    'description': config['description'],
                    'result_directory': result_dir,
                    'status': 'SUCCESS'
                })
                print(f"✓ Model {config['name']} evaluation completed successfully!")
            else:
                results.append({
                    'model_name': config['name'],
                    'description': config['description'],
                    'result_directory': None,
                    'status': 'FAILED'
                })
                print(f"✗ Model {config['name']} evaluation failed!")
                
        except Exception as e:
            print(f"✗ Error evaluating model {config['name']}: {e}")
            results.append({
                'model_name': config['name'],
                'description': config['description'],
                'result_directory': None,
                'status': f'ERROR: {e}'
            })
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    for result in results:
        status_icon = "✓" if result['status'] == 'SUCCESS' else "✗"
        print(f"{status_icon} {result['model_name']}: {result['status']}")
        if result['result_directory']:
            print(f"    Results: {result['result_directory']}")
    
    print("\n" + "=" * 60)
    print("All evaluations completed!")
    
    return results


if __name__ == "__main__":
    results = evaluate_multiple_models()
