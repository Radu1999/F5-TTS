import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.parquet as pq
import os


df = pd.read_csv('./data/ro_tts_pinyin/metadata.csv', header=None, delimiter='|')

# Strip the corpus prefix so paths become "wavs/<file>.wav".
# Mount-point independent: keep everything from the last "wavs/" segment on.
df[0] = df[0].apply(lambda p: p[p.rindex("wavs/"):] if "wavs/" in p else p)

# # Add "wavs/" prefix to each filename
#df[0] = 'wavs/' + df[0]

print(df.head())

# Write back to the CSV file
df.to_csv('./data/ro_tts_pinyin/metadata.csv', header=False, index=False, sep='|')
