from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import Audio, load_dataset
import torch

# load model and processor
processor = WhisperProcessor.from_pretrained("gigant/whisper-medium-romanian")
model = WhisperForConditionalGeneration.from_pretrained("gigant/whisper-medium-romanian")

# load dummy dataset and read soundfiles
ds = load_dataset("mozilla-foundation/common_voice_13_0", "ro", split="test", streaming=True, trust_remote_code=True)
ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
input_speech = next(iter(ds))["audio"]["array"]
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language = "ro", task = "transcribe")
input_features = processor(input_speech, return_tensors="pt", sampling_rate=16_000).input_features
predicted_ids = model.generate(input_features, max_length=448)
# transcription = processor.batch_decode(predicted_ids)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens = True)
print(transcription)