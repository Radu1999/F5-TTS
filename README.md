# Ro-F5TTS: Romanian TTS via Lightweight Input Adaptation

**F5-TTS-RO: Extending F5-TTS to Romanian TTS via Lightweight Input Adaptation**
(Radu-Gabriel Chivereanu, Tiberiu Boros) —
[arXiv:2512.12297](https://arxiv.org/abs/2512.12297), accepted at LREC-COLING 2026.

This repository is a fork of [F5-TTS](https://github.com/SWivid/F5-TTS) implementing the
paper's method: a **frozen F5-TTS backbone + a lightweight input-level adapter** that
extends the text encoder to Romanian while preserving the model's existing capabilities
(voice cloning, English and Chinese). Example audio:
[github.com/racai-ro/Ro-F5TTS](https://github.com/racai-ro/Ro-F5TTS).

## Method

Instead of finetuning the whole model, Ro-F5TTS appends a small sub-network on the text
input side and trains only that, keeping all original weights frozen:

```
ŷ = TTS( ConvNeXt( E_R(x) ) )
```

- `E_R(x)` — a new character-level embedding matrix for the Romanian vocabulary,
  initialized as a copy of the frozen F5-TTS embedding (see `Trainer.train` in
  `src/f5_tts/model/language_trainer.py`).
- `ConvNeXt(·)` — a stack of 4 ConvNeXt-V2 blocks (dim 512) modeling the
  co-dependencies between the character embeddings — a "soft" letter-to-sound layer.

The adapter's output replaces the text embedding fed to the frozen text encoder.
This is `LanguageModule` in `src/f5_tts/model/backbones/dit.py` (checkpoint keys
`transformer.language_module.text_embed.*` and `transformer.language_module.text_blocks.*`).
`src/f5_tts/train/finetune_cli.py` freezes all parameters except
`transformer.language_module.parameters()`.

### Code switching

In mixed Romanian–English text, `~` marks a language switch. The switch token is removed
from the sequence and a cumulative region flag is kept; the adapter processes the whole
sequence (non-Romanian regions are fed a placeholder id) and its output then passes
through the original frozen ConvNeXt text encoder. See `LanguageModule.forward` and
`TextEmbedding.forward` in `src/f5_tts/model/backbones/dit.py`.

## Training

Data: Romanian audio + transcripts (SWARA corpus in the paper), prepared under
`data/ro_tts_pinyin/` with the `pinyin` tokenizer (non-Chinese characters pass through
unchanged) and its `vocab.txt`.

Paper setup (Table 1):

| parameter | value |
|---|---|
| learning rate | 1e-4 |
| batch size | 16384 frames (≤ 128 samples) |
| warmup updates | 50 |
| total updates | 40500 |
| hardware | 1× A100, ~12 h |

`run_training.sh` wraps the exact command:

```bash
export WANDB_API_KEY=<key>   # or: wandb login
accelerate launch src/f5_tts/train/finetune_cli.py --logger wandb --dataset_name ro_tts \
    --num_warmup_updates 50 --last_per_updates 2000 --log_samples \
    --learning_rate 1e-4 --save_per_updates 300 \
    --batch_size_per_gpu 16384 --max_samples 128 --keep_last_n_checkpoints 3 \
    --epochs 50 --checkpoint_path ro_tts_lang_mod
```

Checkpoints are written to `ckpts/ro_tts_lang_mod/` (`model_last.pt` + rotated
`model_<update>.pt`).

> **Note** — the pretrained F5-TTS v1 base checkpoint must be available to the trainer:
> either place it in `ckpts/ro_tts_lang_mod/` as `pretrained_model_1250000.safetensors`,
> or launch with `--finetune --pretrain <path/to/model_1250000.safetensors>` (it will be
> copied into the checkpoint dir as `pretrained_*`). Without it, `language_trainer.py`
> prints "No pretrained model found" and the backbone starts from scratch.

### SLURM (UPB cluster)

`sbatch submit_job.sh` runs `run_training.sh` inside the `tts.sif` Singularity image on
`dgxa100`, with `data/` and `ckpts/` bind-mounted.

## Inference

### CLI

```bash
f5-tts_infer-cli \
    --model F5TTS_v1_Base \
    --ckpt_file ckpts/ro_tts_lang_mod/model_last.pt \
    --vocab_file data/ro_tts_pinyin/vocab.txt \
    --ref_audio "reference.wav" \
    --ref_text "Transcrierea audio de referință." \
    --gen_text "Text de sintetizat, cu span-uri engleze separate de ~."
```

`--ckpt_file` / `--vocab_file` default to exactly the paths above, so after training a
plain `f5-tts_infer-cli --ref_audio ... --ref_text ... --gen_text ...` works.
`model_last.pt` is a full trainer checkpoint; its EMA weights are loaded automatically.

### Gradio web UI

```bash
f5-tts_infer-gradio --port 7860
```

The default model is the local Romanian checkpoint; the E2-TTS option stays available
for English/Chinese.

## Finetuning with / without the ConvNeXt module

The DiT backbone encodes text with a learned embedding plus an **optional stack of
ConvNeXt-V2 blocks** (together with sinusoidal positional embeddings) on top of it — see
`TextEmbedding` in `src/f5_tts/model/backbones/dit.py`. The number of blocks is the
architecture parameter `conv_layers`:

| `conv_layers` | text encoder |
|---|---|
| `4` (all shipped F5TTS configs, the default) | embedding + 4 × ConvNeXt-V2 blocks (+ sinusoidal pos. embedding) |
| `0` | plain embedding only (no ConvNeXt blocks, no text-side positional embedding) |

Note: the code-switching `LanguageModule` used for `lang='ro'` training has its own 4
ConvNeXt-V2 blocks, which are *not* controlled by `conv_layers`.

#### With ConvNeXt (default)

All shipped paths already use `conv_layers=4` — nothing to change. E.g. the CLI trainer
used by `run_training.sh`:

```bash
accelerate launch src/f5_tts/train/finetune_cli.py \
    --exp_name F5TTS_v1_Base --finetune --pretrain <path/to/pretrained.ckpt> \
    --dataset_name ro_tts --epochs 50 --learning_rate 1e-4 \
    --batch_size_per_gpu 16384 --max_samples 128 --logger wandb
```

or the hydra config:

```bash
accelerate launch src/f5_tts/train/train.py --config-name F5TTS_v1_Base.yaml
```

Note that `finetune_cli.py` freezes the whole model except `transformer.language_module`
parameters — that is the part actually trained there.

#### Without ConvNeXt

Build the model without the block stack by setting `conv_layers=0`:

- **hydra / `train.py` path** — override the config:

  ```bash
  accelerate launch src/f5_tts/train/train.py --config-name F5TTS_v1_Base.yaml ++model.arch.conv_layers=0
  ```

  The standard trainer (`src/f5_tts/model/trainer.py`) loads `pretrained_*` checkpoints
  *strictly*, so initializing from the official F5TTS checkpoints will fail on the extra
  `text_blocks.*` tensors — either remove those keys from the checkpoint beforehand, or
  start from scratch.

- **CLI path (`finetune_cli.py`)** — `conv_layers=4` is hardcoded in the `model_cfg`
  dict; set it to `0` there. Its trainer (`src/f5_tts/model/language_trainer.py`) loads
  the pretrained checkpoint with `strict=False`, so the ConvNeXt tensors are silently
  dropped and all remaining weights initialize as usual.

When flipping between the two variants, also change the checkpoint dir
(`++model.name=...` or `--checkpoint_path ...`) so the runs do not share (or overwrite)
the same `ckpts/` folder.

#### Inference

Load a checkpoint with the same `conv_layers` you trained with — checkpoint loading is
strict. For no-ConvNeXt checkpoints pass `conv_layers: 0` in the model cfg of
`load_model(...)`; for the standard ones use `conv_layers: 4`.

## Evaluation

`src/f5_tts/train/compare.py` generates speech from the pretrained and finetuned
checkpoints for the same reference/text pairs (the paper's pre/post comparison).

## Installation

```bash
# Python 3.10 environment
conda create -n f5-tts python=3.10
conda activate f5-tts

# PyTorch for your device, e.g.
pip install torch==2.4.0+cu124 torchaudio==2.4.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124

# install (editable, since training is part of the repo)
pip install -e .
```

## Citation

If you use Ro-F5TTS, please cite:

```bibtex
@inproceedings{chivereanu-boros-2026-f5ts-ro,
  title     = {F5-TTS-RO: Extending F5-TTS to Romanian TTS via Lightweight Input Adaptation},
  author    = {Chivereanu, Radu-Gabriel and Boros, Tiberiu},
  booktitle = {Proceedings of LREC-COLING 2026},
  year      = {2026}
}
```

and the base model:

```bibtex
@article{chen-etal-2024-f5tts,
      title={F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching}, 
      author={Yushen Chen and Zhikang Niu and Ziyang Ma and Keqi Deng and Chunhui Wang and Jian Zhao and Kai Yu and Xie Chen},
      journal={arXiv preprint arXiv:2410.06885},
      year={2024},
}
```

## License

Code is released under MIT License. The pre-trained F5-TTS models are licensed under the
CC-BY-NC license due to the training data Emilia, which is an in-the-wild dataset.
