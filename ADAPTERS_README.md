# F5-TTS Language Adapters

This guide explains how to use the language adapter system in F5-TTS to add Romanian language support while maintaining English capabilities and voice cloning prowess.

## Overview

The adapter system allows you to:
- ✅ Add Romanian language support to F5-TTS
- ✅ Preserve original English capabilities
- ✅ Maintain voice cloning abilities
- ✅ Use efficient parameter updates (only ~2-5% of model parameters)
- ✅ Switch between languages seamlessly

## Architecture

The adapter system uses:
- **LoRA (Low-Rank Adaptation)**: Efficient parameter-efficient fine-tuning
- **Bottleneck Adapters**: Alternative adapter architecture 
- **Language-specific routing**: Automatic language detection and routing
- **Extended vocabulary**: Support for Romanian diacritics
- **Voice preservation**: Maintains speaker characteristics across languages

## Installation

The adapter functionality is built into the main F5-TTS codebase. Ensure you have the required dependencies:

```bash
pip install torch torchaudio
pip install safetensors
pip install accelerate
```

## Quick Start

### 1. Prepare Romanian Dataset

Create a Romanian dataset in the same format as F5-TTS training data:

```bash
# Organize your Romanian audio files and transcripts
mkdir -p data/romanian_dataset
# Place your .wav files and corresponding .txt files in the directory
```

### 2. Train Romanian Adapters

```bash
python src/f5_tts/train/train_adapters.py \
    --base_model_path path/to/pretrained/f5tts_model.safetensors \
    --romanian_dataset data/romanian_dataset \
    --output_dir adapters/romanian \
    --adapter_type lora \
    --adapter_rank 16 \
    --adapter_alpha 16.0 \
    --learning_rate 1e-4 \
    --batch_size 16 \
    --epochs 20
```

### 3. Use for Inference

```bash
python src/f5_tts/infer/infer_adapters.py \
    --ref_audio path/to/reference.wav \
    --ref_text "Aceasta este vocea de referință" \
    --gen_text "Generează această propoziție în română" \
    --output generated_romanian.wav \
    --base_ckpt path/to/base_model.safetensors \
    --adapter_ckpt adapters/romanian/model_best.pt \
    --language ro
```

## Detailed Usage

### Training Configuration

The adapter training script supports various configurations:

```bash
python src/f5_tts/train/train_adapters.py \
    --base_model_path path/to/base/model.safetensors \
    --romanian_dataset path/to/romanian/data \
    --output_dir path/to/adapter/output \
    --adapter_type lora \              # or "bottleneck"
    --adapter_rank 16 \                # LoRA rank (higher = more capacity)
    --adapter_alpha 16.0 \             # LoRA scaling factor
    --adapter_dropout 0.1 \            # Dropout rate
    --learning_rate 1e-4 \             # Learning rate
    --batch_size 16 \                  # Batch size
    --epochs 20 \                      # Training epochs
    --warmup_steps 1000 \              # Warmup steps
    --save_steps 1000 \                # Save frequency
    --mixed_precision fp16             # Mixed precision training
```

### Inference Options

The inference script provides comprehensive control:

```bash
python src/f5_tts/infer/infer_adapters.py \
    --ref_audio reference.wav \
    --ref_text "Text of reference audio" \
    --gen_text "Text to generate" \
    --output output.wav \
    --base_ckpt base_model.safetensors \
    --adapter_ckpt adapter_model.pt \
    --language ro \                    # "en" or "ro"
    --steps 32 \                       # Sampling steps
    --cfg_strength 2.0 \               # Guidance strength
    --speed 1.0 \                      # Speech speed
    --seed 42 \                        # Random seed
    --remove_silence                   # Remove silence
```

## Programming Interface

### Using Adapters in Code

```python
from f5_tts.model.adapters import AdapterConfig
from f5_tts.infer.infer_adapters import F5TTSAdapters

# Configure adapters
adapter_config = AdapterConfig(
    languages={
        "en": {"adapter_type": "lora", "rank": 16, "alpha": 16.0},
        "ro": {"adapter_type": "lora", "rank": 16, "alpha": 16.0}
    },
    default_language="en",
    enable_text_adapter=True,
    enable_attention_adapter=True,
    enable_feedforward_adapter=True,
)

# Initialize F5-TTS with adapters
tts = F5TTSAdapters(
    model="F5TTS_Base",
    ckpt_file="path/to/base_model.safetensors",
    adapter_ckpt_file="path/to/adapter_model.pt",
    adapter_config=adapter_config,
    languages=["en", "ro"],
    default_language="en",
)

# Generate Romanian speech
audio, seed = tts.infer(
    ref_file="reference.wav",
    ref_text="Aceasta este vocea de referință",
    gen_text="Generează această propoziție în română",
    language="ro",
    file_wave="output_romanian.wav"
)

# Generate English speech (preserves original capabilities)
audio, seed = tts.infer(
    ref_file="reference.wav",
    ref_text="This is the reference voice",
    gen_text="Generate this sentence in English",
    language="en",
    file_wave="output_english.wav"
)
```

### Custom Model Integration

```python
from f5_tts.model import CFM, DiT
from f5_tts.model.adapters import AdapterConfig

# Create adapter configuration
adapter_config = AdapterConfig(
    languages={
        "en": {"adapter_type": "lora", "rank": 16, "alpha": 16.0},
        "ro": {"adapter_type": "lora", "rank": 16, "alpha": 16.0}
    },
    default_language="en"
)

# Create model with adapters
model = CFM(
    transformer=DiT(
        dim=1024,
        depth=22,
        heads=16,
        ff_mult=2,
        text_dim=512,
        conv_layers=4,
        adapter_config=adapter_config,  # Add adapter support
        text_num_embeds=vocab_size,
        mel_dim=100
    ),
    mel_spec_kwargs=mel_spec_kwargs,
    adapter_config=adapter_config,
)

# Use for training or inference with language specification
output = model(
    inp=mel_input,
    text=text_input,
    language="ro",  # Specify target language
    language_ids=language_tensor  # Or use tensor of language IDs
)
```

## Dataset Preparation

### Romanian Text Processing

The system includes Romanian-specific text preprocessing:

```python
# Automatic diacritic normalization
# ş, ţ → ș, ț (proper Romanian diacritics)
# Handles both uppercase and lowercase

# Example texts for training:
romanian_texts = [
    "Acesta este un exemplu de text românesc cu caractere specifice: ă, â, î, ș, ț.",
    "Generarea vocii în limba română menține caracteristicile vorbitorului original.",
    "Sistemul de adaptoare permite transferul abilităților de clonare vocală."
]
```

### Data Format

Prepare your Romanian dataset in this structure:

```
romanian_dataset/
├── metadata.csv          # Optional: audio_path,text,speaker_id
├── audio/
│   ├── speaker1_001.wav
│   ├── speaker1_002.wav
│   └── ...
└── transcripts/
    ├── speaker1_001.txt
    ├── speaker1_002.txt
    └── ...
```

Or use the metadata.csv format:
```csv
audio_path,text,duration,speaker_id
audio/speaker1_001.wav,"Acesta este primul text.",3.45,speaker1
audio/speaker1_002.wav,"Acesta este al doilea text.",2.87,speaker1
```

## Technical Details

### Adapter Architecture

The system implements several adapter types:

1. **LoRA Adapters**: 
   - Low-rank decomposition: `Δ = BA`
   - Efficient parameter updates
   - Configurable rank and scaling

2. **Bottleneck Adapters**:
   - Down-projection → Activation → Up-projection
   - Residual connections
   - Configurable bottleneck dimension

3. **Multi-Language Routing**:
   - Language-specific adapter selection
   - Automatic language embedding
   - Batch-level language mixing support

### Memory and Performance

- **Parameter Efficiency**: Adapters add only 2-5% of original parameters
- **Memory Usage**: Minimal overhead during inference
- **Training Speed**: Faster than full fine-tuning
- **Quality**: Maintains voice cloning capabilities

### Language Support

Currently supported:
- ✅ English (en) - Original capabilities preserved
- ✅ Romanian (ro) - Full adapter support

Easily extensible to other languages by:
1. Adding language-specific character mappings
2. Creating new adapter configurations
3. Training on target language data

## Troubleshooting

### Common Issues

1. **Vocabulary Size Mismatch**
   ```python
   # Ensure Romanian characters are included in vocabulary
   romanian_chars = ['ă', 'â', 'î', 'ș', 'ț', 'Ă', 'Â', 'Î', 'Ș', 'Ț']
   ```

2. **Adapter Loading Errors**
   ```python
   # Check adapter checkpoint contains correct parameter names
   # Parameters should include: language_adapters, attention_adapter, etc.
   ```

3. **Language Switching**
   ```python
   # Ensure language parameter is correctly passed
   tts.infer(..., language="ro")  # Explicit language specification
   ```

### Performance Optimization

```python
# Use mixed precision for faster training
accelerate launch --mixed_precision=fp16 train_adapters.py ...

# Optimize batch size for your GPU memory
--batch_size 8   # Reduce if out of memory
--batch_size 32  # Increase for better utilization

# Use checkpointing for large models
--checkpoint_activations  # Saves memory at cost of speed
```

## Contributing

To add support for additional languages:

1. Create character mappings for the target language
2. Prepare training data in the required format
3. Train adapters using the provided scripts
4. Test inference and quality
5. Submit a pull request with your changes

## Citation

If you use this adapter system, please cite the original F5-TTS paper and mention the adapter extension:

```bibtex
@article{f5tts,
  title={F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching},
  author={...},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This adapter system is released under the same license as F5-TTS. See the main repository for details. 