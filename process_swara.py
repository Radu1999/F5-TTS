import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.parquet as pq
import os


df = pd.read_csv('./data/ro_tts_pinyin/metadata.csv', header=None, delimiter='|')

# Remove the prefix from the first column
prefix_to_remove = '/media/DATA/CORPORA/SWARA2.0/SWARA2.0_22k/'
df[0] = df[0].str.replace(prefix_to_remove, '', regex=False)
# prefix_to_remove = 'wavs//media/DATA/CORPORA/SWARA2.0/SWARA1.0_22k/'
# df[0] = df[0].str.replace(prefix_to_remove, '', regex=False)

# # Add "wavs/" prefix to each filename
#df[0] = 'wavs/' + df[0]

print(df.head())

# Write back to the CSV file
df.to_csv('./data/ro_tts_pinyin/metadata.csv', header=False, index=False, sep='|')
