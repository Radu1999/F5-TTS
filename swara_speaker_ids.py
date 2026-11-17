from datasets import Dataset
from datasets.arrow_writer import ArrowWriter
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Paths
arrow_path = "data/ro_tts_pinyin/raw.arrow"
output_arrow_path = "data/ro_tts_pinyin/raw_with_client_id.arrow"

# Load Arrow dataset and convert to pandas
ds = Dataset.from_file(arrow_path)
df = ds.to_pandas()

# Extract speaker from audio_path (assuming format like "wavs/speaker_filename.wav")
df['speaker'] = df['audio_path'].str.extract(r'wavs/([^_]+)_')

# Fill any missing speakers with 'unknown'
df['speaker'] = df['speaker'].fillna('unknown')

# Convert dataframe to records for ArrowWriter
records = df.to_dict(orient="records")

with ArrowWriter(path=Path(output_arrow_path).as_posix(), writer_batch_size=100) as writer:
    for record in tqdm(records, desc="Writing to raw_with_client_id.arrow ..."):
        writer.write(record)

print(f"Dataset saved to {output_arrow_path}")
print(f"Number of distinct speakers: {df['speaker'].nunique()}")
print(f"Speakers: {df['speaker'].unique()}")
print(f"Sample data:\n{df[['audio_path', 'speaker']].head()}")

# Read the written arrow file back in to test
print("\nTesting read-back of written Arrow file:")
ds_test = Dataset.from_file(output_arrow_path)
df_test = ds_test.to_pandas()
print(f"Read-back: Number of rows: {len(df_test)}")
print(f"Read-back: Number of distinct speakers: {df_test['speaker'].nunique()}")
print(f"Read-back: Speakers: {df_test['speaker'].unique()}")
print(f"Read-back: Sample data:\n{df_test[['audio_path', 'speaker']].head()}")

