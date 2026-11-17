import pandas as pd
from collections import defaultdict
import os

def group_entries_by_identity(csv_path):
    
    # Read CSV without header since it appears to be just data
    df = pd.read_csv(csv_path, header=None, sep='|', names=['file_path', 'text'])
    
    print(f"Total entries: {len(df)}")
    
    # Group by identity
    grouped_data = defaultdict(list)
    
    for index, row in df.iterrows():
        file_path = row['file_path']
        text = row['text']
        
        # Extract identity from file path: wavs/identity_rnd#_###.wav
        filename = os.path.basename(file_path)
        # Remove .wav extension and split by underscore
        parts = filename.replace('.wav', '').split('_')
        
        if len(parts) >= 2:
            identity = parts[0]  # First part is the identity
            grouped_data[identity].append({
                'file_path': file_path,
                'text': text,
                'round': parts[1] if len(parts) > 1 else 'unknown',
                'number': parts[2] if len(parts) > 2 else 'unknown'
            })
    
    return grouped_data

def display_grouping_stats(grouped_data):
    """Display statistics about the grouped data"""
    print("\n=== GROUPING STATISTICS ===")
    print(f"Total number of identities: {len(grouped_data)}")
    
    for identity, entries in grouped_data.items():
        print(f"\nIdentity: {identity}")
        print(f"  Total entries: {len(entries)}")
        
        # Count by rounds
        rounds = defaultdict(int)
        for entry in entries:
            rounds[entry['round']] += 1
        
        print("  Entries by round:")
        for round_name, count in sorted(rounds.items()):
            print(f"    {round_name}: {count}")

def save_grouped_data(grouped_data, output_dir="grouped_output"):
    """Save grouped data to separate CSV files"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n=== SAVING GROUPED DATA TO {output_dir} ===")
    
    for identity, entries in grouped_data.items():
        output_file = os.path.join(output_dir, f"{identity}_metadata.csv")
        
        # Convert to DataFrame
        df = pd.DataFrame(entries)
        df.to_csv(output_file, index=False)
        
        print(f"Saved {len(entries)} entries for identity '{identity}' to {output_file}")

if __name__ == "__main__":
    # Path to the metadata.csv file
    csv_path = "data/ro_tts_pinyin/metadata.csv"
    
    try:
        # Group entries by identity
        grouped_data = group_entries_by_identity(csv_path)
        
        # Display statistics
        display_grouping_stats(grouped_data)
        
        # Save grouped data to separate files
        save_grouped_data(grouped_data)
        
        print("\n=== SAMPLE DATA ===")
        # Show a sample of each identity
        for identity, entries in list(grouped_data.items())[:3]:  # Show first 3 identities
            print(f"\nSample entries for identity '{identity}':")
            for i, entry in enumerate(entries[:3]):  # Show first 3 entries
                print(f"  {i+1}. {entry['file_path']} -> {entry['text'][:50]}...")
            if len(entries) > 3:
                print(f"  ... and {len(entries) - 3} more entries")
        
    except FileNotFoundError:
        print(f"Error: Could not find the file {csv_path}")
    except Exception as e:
        print(f"Error: {e}")
