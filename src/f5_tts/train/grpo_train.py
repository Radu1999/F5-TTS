# train_grpo.py
from f5_tts.model.dataset import load_dataset
from trl import GRPOConfig, GRPOTrainer
from f5_tts.model import CFM, DiT
from f5_tts.model.utils import get_tokenizer, convert_char_to_pinyin
import torch
from safetensors.torch import load_file
from f5_tts.infer.utils_infer import cfg_strength, load_vocoder, nfe_step, sway_sampling_coef
import torchaudio
import torch.nn.functional as F
from f5_tts.infer.utils_infer import load_checkpoint
import argparse
from transformers import AutoModelForCausalLM


class PerceptualLoss(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.features = {}

        # Register hooks for all convnext blocks
        for idx, module in enumerate(backbone.convnext):
            module.register_forward_hook(self.save_output(f"convnext.{idx}"))

    def save_output(self, name):
        def hook(module, input, output):
            self.features[name] = output

        return hook

    def forward(self, x, y):
        self.features.clear()
        _ = self.backbone(x)
        feats_x = self.features.copy()

        self.features.clear()
        _ = self.backbone(y)
        feats_y = self.features.copy()

        # Average L1 loss across all hooked layers
        losses = []
        for layer in feats_x.keys():
            losses.append(F.l1_loss(feats_x[layer], feats_y[layer]))

        return sum(losses) / len(losses)


vocoder = load_vocoder(
    vocoder_name='vocos', is_local=False
)

# -------------------------- Dataset Settings --------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_path', type=str, default=r"C:\Users\Mihaitza\Desktop\F5-TTS\ckpts\ro_tts\pretrained_model_1250000.safetensors", help='Path to checkpoint file (.pt or .safetensors)')
parser.add_argument('--dataset_name', type=str, default='ro_tts', help='Dataset name to load')
args = parser.parse_args()

target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
win_length = 1024
n_fft = 1024
mel_spec_type = "vocos"  # 'vocos' or 'bigvgan'
dataset_name = args.dataset_name
tokenizer = "pinyin"

model_cls = DiT
model_cfg = dict(
    dim=1024,
    depth=22,
    heads=16,
    ff_mult=2,
    text_dim=512,
    conv_layers=4,
)

mel_spec_kwargs = dict(
    n_fft=n_fft,
    hop_length=hop_length,
    win_length=win_length,
    n_mel_channels=n_mel_channels,
    target_sample_rate=target_sample_rate,
    mel_spec_type=mel_spec_type,
)

vocab_char_map, vocab_size = get_tokenizer(dataset_name, tokenizer)

model = CFM(
    transformer=model_cls(**model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
    mel_spec_kwargs=mel_spec_kwargs,
    vocab_char_map=vocab_char_map,
)

if args.ckpt_path:
    model = load_checkpoint(model, ckpt_path=args.ckpt_path, device="cuda")

train_dataset = load_dataset(dataset_name)
train_dataset.data = train_dataset.data.add_column("prompt", [0] * len(train_dataset))

prompt = 'Generate the transliteration (write the sentence in English characters, maintaining pronunciation) of the ' \
         'following Romanian sentence into English:"{}". Return just the transliteration. '


def compute_prompt(example):
    return prompt.format(''.join(example['text']))


train_dataset.data = train_dataset.data.map(lambda x: {"prompt": compute_prompt(x)})
loss = PerceptualLoss(vocoder.backbone)
global_step = 0


# Define the reward function, which rewards completions that are close to 20 characters
def reward_gen(completions, mel_spec, **kwargs):
    global global_step
    spec = mel_spec[0].permute(1, 0)
    ref_audio_len = spec.shape[0]
    text_inputs = [list(completion) * 2 for completion in completions]

    cond = spec[:ref_audio_len].unsqueeze(0).to('cuda', dtype=torch.float32).repeat(len(text_inputs), 1, 1)
    generated, _ = model.sample(
        cond=cond,
        text=text_inputs,
        duration=ref_audio_len * 2,
        steps=nfe_step,
        cfg_strength=cfg_strength,
        sway_sampling_coef=sway_sampling_coef,
    )

    ref_mel_spec = spec.unsqueeze(0).permute(0, 2, 1).to('cuda', dtype=torch.float32)
    rewards = []
    for gen in generated:
        gen_mel_spec = gen[ref_audio_len:, :].unsqueeze(0).permute(0, 2, 1).to('cuda', dtype=torch.float32)
        
        # Ensure both mel spectrograms have the same time dimension
        min_time_len = min(gen_mel_spec.shape[2], ref_mel_spec.shape[2])
        gen_mel_spec = gen_mel_spec[:, :, :min_time_len]
        ref_mel_spec_cropped = ref_mel_spec[:, :, :min_time_len]
        
        reward = -loss(gen_mel_spec, ref_mel_spec_cropped)
        rewards.append(float(reward.detach().cpu().item()))

    global_step += 1
    if global_step % 1000 == 0:
        gen_mel_spec_sanity = generated[0][ref_audio_len:, :].unsqueeze(0).permute(0, 2, 1).to('cuda', dtype=torch.float32)
        
        # Ensure both have same length for consistent comparison
        min_time_len_sanity = min(gen_mel_spec_sanity.shape[2], ref_mel_spec.shape[2])
        gen_mel_spec_sanity = gen_mel_spec_sanity[:, :, :min_time_len_sanity]
        ref_mel_spec_sanity = ref_mel_spec[:, :, :min_time_len_sanity]
        
        gen_audio = vocoder.decode(gen_mel_spec_sanity).cpu()
        ref_audio = vocoder.decode(ref_mel_spec_sanity).cpu()
        torchaudio.save(f"./sanity_check_grpo_gen_{global_step}.wav", gen_audio, target_sample_rate)
        torchaudio.save(f"./sanity_check_grpo_ref_{global_step}.wav", ref_audio, target_sample_rate)
    return rewards


training_args = GRPOConfig(output_dir="gemma2b")

llm = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b-it",
    device_map="auto",
    attn_implementation='eager'
)

trainer = GRPOTrainer(
    model=llm,
    reward_funcs=reward_gen,
    args=training_args,
    train_dataset=train_dataset,
)
trainer.train()
