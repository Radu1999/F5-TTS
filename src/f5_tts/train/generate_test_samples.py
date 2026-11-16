import pandas as pd
import random
from collections import defaultdict
from pydub import AudioSegment
import os


def extract_speakers(threshold_db=-30):
    """
    Extract 10 speakers from validated.tsv with the following criteria:
    - 0 downvotes
    - Distributed across age groups: twenties, thirties, fourties, fifties
    - Mixed between male and female
    - Only include audio files with RMS amplitude above threshold_db
    """

    tsv_path = r"C:\Users\Mihaitza\Desktop\F5-TTS\data\ro_tts_common_pinyin\cv-corpus-22.0-2025-06-20\ro\validated.tsv"

    print("Reading validated.tsv file...")
    df = pd.read_csv(tsv_path, sep='\t')
    print(f"Total records: {len(df)}")

    # Filter for 0 downvotes
    df_filtered = df[df['down_votes'] == 0].copy()
    print(f"Records with 0 downvotes: {len(df_filtered)}")

    # Target age groups
    target_ages = ['twenties', 'thirties', 'fourties', 'fifties']
    df_filtered = df_filtered[
        (df_filtered['age'].isin(target_ages)) &
        (df_filtered['gender'].notna()) &
        (df_filtered['gender'] != '')
        ].copy()
    print(f"Records with target ages and valid gender: {len(df_filtered)}")

    # Group by age and gender
    grouped = defaultdict(list)
    for _, row in df_filtered.iterrows():
        age = row['age']
        gender = row['gender']
        if gender == 'male_masculine':
            gender_normalized = 'male'
        elif gender == 'female_feminine':
            gender_normalized = 'female'
        else:
            continue

        grouped[(age, gender_normalized)].append({
            'client_id': row['client_id'],
            'path': row['path'],
            'age': age,
            'gender': gender_normalized,
            'sentence': row['sentence']
        })

    # Function to check amplitude
    def is_loud_enough(file_path, threshold_db):
        if not os.path.exists(file_path):
            return False
        audio = AudioSegment.from_file(file_path)
        return audio.dBFS >= threshold_db

    # Select speakers
    selected_speakers = []
    for age in target_ages:
        for gender in ['male', 'female']:
            candidates = grouped.get((age, gender), [])
            # Shuffle to randomize selection
            random.shuffle(candidates)
            for cand in candidates:
                if is_loud_enough(cand['path'], threshold_db):
                    selected_speakers.append(cand)
                    break  # Only one per age/gender first
        if len(selected_speakers) >= 8:  # 2 per age group
            break

    # Fill remaining slots from any available loud audio
    all_remaining = []
    for (age, gender), speakers in grouped.items():
        for speaker in speakers:
            path = r"C:\Users\Mihaitza\Desktop\F5-TTS\data\ro_tts_common_pinyin\wavs"
            if speaker not in selected_speakers and is_loud_enough(path + "\\" +  speaker['path'].split('.')[0] + '.wav', threshold_db):
                all_remaining.append(speaker)

    while len(selected_speakers) < 10 and all_remaining:
        speaker = random.choice(all_remaining)
        selected_speakers.append(speaker)
        all_remaining.remove(speaker)

    # Display results
    print(f"\nSelected {len(selected_speakers)} speakers:")
    print("-" * 80)

    age_gender_count = defaultdict(int)
    for i, speaker in enumerate(selected_speakers, 1):
        print(f"{i:2d}. Client ID: {speaker['client_id'][:16]}...")
        print(f"    Age: {speaker['age']:<10} Gender: {speaker['gender']:<6}")
        print(f"    Path: {speaker['path']}")
        print(f"    Sentence: {speaker['sentence']}\n")
        age_gender_count[(speaker['age'], speaker['gender'])] += 1

    print("\nDistribution summary:")
    for (age, gender), count in sorted(age_gender_count.items()):
        print(f"  {age} {gender}: {count}")

    # Save to CSV
    output_df = pd.DataFrame(selected_speakers)
    output_path = "selected_speakers.csv"
    output_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    return selected_speakers


if __name__ == "__main__":
    random.seed(42)
    speakers = extract_speakers(threshold_db=-40)
