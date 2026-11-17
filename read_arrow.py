from datasets import Dataset
from datasets.arrow_writer import ArrowWriter
import pandas as pd
from pathlib import Path
from tqdm import tqdm

arrow_path = "data/ro_tts_common_pinyin/raw.arrow"

tsv_path = "data/ro_tts_common_pinyin/validated.tsv"
# Load Arrow dataset
ds = Dataset.from_file(arrow_path)

# Load validated.tsv
df = pd.read_csv(tsv_path, sep="\t")

# Add prefix so it matches audio_path in arrow
prefix = "data/ro_tts_common_pinyin/wavs/"
df["path"] = df["path"].str.replace(".mp3", ".wav", regex=False)
df["audio_path"] = prefix + df["path"]

# Keep only columns needed for join
df = df[["audio_path", "client_id"]]

num_distinct_clients = df["client_id"].nunique()
print("Number of distinct client IDs:", num_distinct_clients)
print(df.head(10)[["audio_path", "client_id"]])

# Convert Arrow dataset to pandas
ds_df = ds.to_pandas()
print(ds_df.head(10))

# Merge on audio_path
merged = ds_df.merge(df, on="audio_path", how="left")

# Back to Arrow dataset
ds_merged = Dataset.from_pandas(merged)
output_arrow_path = "data/ro_tts_common_pinyin/raw_with_client_id.arrow"
print(ds_merged)
print(ds_merged[0])

records = merged.to_dict(orient="records")

# Write to a single .arrow file
with ArrowWriter(path=Path(output_arrow_path).as_posix(), writer_batch_size=100) as writer:
    for record in tqdm(records, desc="Writing to raw_with_client_id.arrow ..."):
        writer.write(record)