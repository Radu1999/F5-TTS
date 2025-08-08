import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from safetensors.torch import load_file
from f5_tts.model import CFM, DiT
from f5_tts.model.utils import get_tokenizer
from f5_tts.model.dataset import load_dataset, collate_fn
from torch.utils.data import DataLoader

nfe_step = 32  # 16, 32
cfg_strength = 2.0
sway_sampling_coef = -1.0

target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
win_length = 1024
n_fft = 1024
mel_spec_type = "vocos"  # 'vocos' or 'bigvgan'

mel_spec_kwargs = dict(
    n_fft=n_fft,
    hop_length=hop_length,
    win_length=win_length,
    n_mel_channels=n_mel_channels,
    target_sample_rate=target_sample_rate,
    mel_spec_type=mel_spec_type,
)

model_cls = DiT
model_cfg = dict(
    dim=1024,
    depth=22,
    heads=16,
    ff_mult=2,
    text_dim=512,
    conv_layers=4,
)

vocab_char_map, vocab_size = get_tokenizer('Emilia_ZH_EN', 'pinyin')
model_pre = CFM(
    transformer=model_cls(**model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
    mel_spec_kwargs=mel_spec_kwargs,
    vocab_char_map=vocab_char_map,
)

checkpoint_pre = load_file(r'C:\Users\Mihaitza\Desktop\F5-TTS\checkpoints\model_1250000.safetensors')
checkpoint_pre = {"ema_model_state_dict": checkpoint_pre}

checkpoint_pre["model_state_dict"] = {
    k.replace("ema_model.", ""): v
    for k, v in checkpoint_pre["ema_model_state_dict"].items()
    if k not in ["initted", "update", "step"]
}
model_pre.load_state_dict(checkpoint_pre['model_state_dict'])


vocab_char_map, vocab_size = get_tokenizer('ro_tts', 'pinyin')
model_ft = CFM(
    transformer=model_cls(**model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
    mel_spec_kwargs=mel_spec_kwargs,
    vocab_char_map=vocab_char_map,
)
checkpoint_ft = torch.load(r'C:\Users\Mihaitza\Desktop\F5-TTS\ckpts\ro_tts_common\model_40000.pt')
model_ft.load_state_dict(checkpoint_ft['model_state_dict'])


dataset = load_dataset('ro_tts')
print(dataset[0])

dataloader = DataLoader(
    dataset,
    collate_fn=collate_fn,
    num_workers=0,
    pin_memory=True,
    batch_size=1,
    shuffle=False,
    generator=None,
)

batch = iter(dataloader).__next__()
text_inputs = batch["text"]
mel_spec = batch["mel"].permute(0, 2, 1)
mel_lengths = batch["mel_lengths"]

ref_audio_len = mel_lengths[0]
infer_text = [
    text_inputs[0] + ([" "] if isinstance(text_inputs[0], list) else " ") + text_inputs[0]
]

# Sample from pre-trained model
generated, trajectory = model_pre.sample(
    cond=mel_spec[0][:ref_audio_len].unsqueeze(0),
    text=infer_text,
    duration=ref_audio_len * 2,
    steps=nfe_step,
    cfg_strength=cfg_strength,
    sway_sampling_coef=sway_sampling_coef,
)

generated = generated.to(torch.float32)
gen_mel_spec = generated[:, ref_audio_len:, :].permute(0, 2, 1)

# Sample from fine-tuned model
generated_ft, trajectory_ft = model_ft.sample(
    cond=mel_spec[0][:ref_audio_len].unsqueeze(0),
    text=infer_text,
    duration=ref_audio_len * 2,
    steps=nfe_step,
    cfg_strength=cfg_strength,
    sway_sampling_coef=sway_sampling_coef,
)

generated_ft = generated_ft.to(torch.float32)
gen_mel_spec_ft = generated_ft[:, ref_audio_len:, :].permute(0, 2, 1)

ref_mel_spec = batch["mel"][0].unsqueeze(0)

# Plot mel spectrograms and trajectory
fig, axes = plt.subplots(2, 3, figsize=(20, 10))

# Plot reference mel spectrogram (full reference from dataset)
ref_mel_np = ref_mel_spec[0].cpu().numpy()
im1 = axes[0, 0].imshow(ref_mel_np, aspect='auto', origin='lower', cmap='viridis')
axes[0, 0].set_title('Reference Mel Spectrogram (Full)')
axes[0, 0].set_xlabel('Time')
axes[0, 0].set_ylabel('Mel Channels')
plt.colorbar(im1, ax=axes[0, 0])

# Plot only the generated parts for comparison (pre-trained)
gen_mel_np = gen_mel_spec[0].cpu().numpy()
im4 = axes[1, 0].imshow(gen_mel_np, aspect='auto', origin='lower', cmap='viridis')
axes[1, 0].set_title('Generated (Pre-trained)')
axes[1, 0].set_xlabel('Time')
axes[1, 0].set_ylabel('Mel Channels')
plt.colorbar(im4, ax=axes[1, 0])

# Plot only the generated parts for comparison (fine-tuned)
gen_mel_ft_np = gen_mel_spec_ft[0].cpu().numpy()
im5 = axes[1, 1].imshow(gen_mel_ft_np, aspect='auto', origin='lower', cmap='viridis')
axes[1, 1].set_title('Generated (Fine-tuned)')
axes[1, 1].set_xlabel('Time')
axes[1, 1].set_ylabel('Mel Channels')
plt.colorbar(im5, ax=axes[1, 1])

# Plot trajectory comparison as vector field
if trajectory is not None and trajectory_ft is not None:
    # Plot trajectory norm comparison only
    trajectory_np = trajectory.cpu().numpy()
    trajectory_ft_np = trajectory_ft.cpu().numpy()

    if len(trajectory_np.shape) == 4:  # [steps, batch, time, mel_dim]
        trajectory_norms = np.linalg.norm(trajectory_np[:, 0, :, :], axis=(1, 2))
        trajectory_ft_norms = np.linalg.norm(trajectory_ft_np[:, 0, :, :], axis=(1, 2))

        axes[1, 2].plot(trajectory_norms, label='Pre-trained', alpha=0.8, linewidth=2)
        axes[1, 2].plot(trajectory_ft_norms, label='Fine-tuned', alpha=0.8, linewidth=2)
        axes[1, 2].set_title('Trajectory Norm Comparison')
        axes[1, 2].set_xlabel('Sampling Step')
        axes[1, 2].set_ylabel('L2 Norm')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
    else:
        # If trajectory has different shape, plot what we can
        axes[1, 2].plot(trajectory_np.flatten()[:1000], label='Pre-trained', alpha=0.8)
        axes[1, 2].plot(trajectory_ft_np.flatten()[:1000], label='Fine-tuned', alpha=0.8)
        axes[1, 2].set_title('Trajectory Values Comparison')
        axes[1, 2].set_xlabel('Index')
        axes[1, 2].set_ylabel('Value')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
else:
    axes[1, 2].text(0.5, 0.5, 'No trajectory data available',
                    horizontalalignment='center', verticalalignment='center', transform=axes[1, 2].transAxes)
    axes[1, 2].set_title('Trajectory Comparison')

plt.tight_layout()

# Save the figure instead of showing it to avoid display issues
try:
    plt.savefig('mel_spectrograms_comparison_ood.png', dpi=300, bbox_inches='tight')
    print("✅ Plot saved as 'mel_spectrograms_comparison.png'")
    print("📁 Plot saved to file since using non-interactive backend")
except Exception as e:
    print(f"❌ Error saving plot: {e}")

plt.close()  # Clean up memory

# Print some statistics
print(f"Reference mel shape: {ref_mel_spec.shape}")
print(f"Generated mel shape (pre-trained): {gen_mel_spec.shape}")
print(f"Generated mel shape (fine-tuned): {gen_mel_spec_ft.shape}")
print(f"Full generated shape (pre-trained): {generated.shape}")
print(f"Full generated shape (fine-tuned): {generated_ft.shape}")
if trajectory is not None:
    print(f"Trajectory shape (pre-trained): {trajectory.shape}")
if trajectory_ft is not None:
    print(f"Trajectory shape (fine-tuned): {trajectory_ft.shape}")
print(f"Reference audio length: {ref_audio_len}")

