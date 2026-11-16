import csv
from ro_diacritics import restore_diacritics

def prepare_dataset(batch):
    """Function to preprocess the dataset with the .map method"""
    transcription = batch["sentence"]
    
    if transcription.startswith('"') and transcription.endswith('"'):
        # we can remove trailing quotation marks as they do not affect the transcription
        transcription = transcription[1:-1]
    
    if transcription[-1] not in [".", "?", "!"]:
        # append a full-stop to sentences that do not end in punctuation
        transcription = transcription + "."
    
    # Check if diacritics restoration is successful
    if restore_diacritics(transcription) is False:
        return None
    
    batch["sentence"] = transcription
    
    return batch

if __name__ == "__main__":
    ds_metadata = []
    speakers = set()
    filtered_count = 0
    with open(r"./validated.tsv", 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter='\t')

        # Print header to see available columns
        print("Available columns:", reader.fieldnames)

        # Read and process each row
        for i, row in enumerate(reader):
            if int(row['down_votes']) == 0:
                # Apply preprocessing to the sentence
                processed_row = prepare_dataset(row)
                if processed_row is not None:
                    ds_metadata.append(['wavs/' + row['path'].replace('.mp3', '.wav'), processed_row['sentence'], row['client_id']])
                    speakers.add(row['client_id'])
                else:
                    filtered_count += 1

    print(f"Processed {len(ds_metadata)} entries")
    print(f"Filtered out {filtered_count} entries where restore_diacritics returned false")
    print(f"Speaker count: {len(speakers)}")

    
    # Write to metadata.csv
    with open(r'C:\Users\Mihaitza\Desktop\F5-TTS\data\ro_tts_common_pinyin\metadata.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter="|")
        
        # Write data
        writer.writerows(ds_metadata)
    
    print(f"Successfully wrote {len(ds_metadata)} entries to metadata.csv")



