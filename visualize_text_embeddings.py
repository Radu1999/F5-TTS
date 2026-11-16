import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap
from datasets import Dataset as Dataset_
from pathlib import Path
import json
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

from f5_tts.model import CFM, DiT
from f5_tts.model.utils import get_tokenizer, list_str_to_idx


def load_romanian_model():
    """Load the trained Romanian F5-TTS model"""
    print("Loading Romanian F5-TTS model...")

    # Model configuration for F5TTS_v1_Base (from setting.json)
    model_cfg = {
        'dim': 1024,
        'depth': 22,
        'heads': 16,
        'ff_mult': 2,
        'text_dim': 512,
        'conv_layers': 4
    }

    # Paths
    ckpt_path = r"checkpoints\swara_te_40800.pt"
    vocab_file = r"data\ro_tts_pinyin\vocab.txt"

    # Load tokenizer
    vocab_char_map, vocab_size = get_tokenizer("ro_tts", "pinyin")

    print(f"Vocab size: {vocab_size}")

    # Create model
    model = CFM(
        transformer=DiT(
            **model_cfg,
            text_num_embeds=vocab_size,
            mel_dim=100
        ),
        vocab_char_map=vocab_char_map
    )

    # Load model weights
    if ckpt_path.endswith('.safetensors'):
        from safetensors.torch import load_file
        checkpoint = load_file(ckpt_path, device="cpu")

        # Handle different checkpoint formats
        if 'ema_model_state_dict' in checkpoint:
            state_dict = checkpoint['ema_model_state_dict']
        else:
            state_dict = checkpoint

        # Remove ema_model prefix if present
        state_dict = {k.replace("ema_model.", ""): v for k, v in state_dict.items()
                      if not k.endswith(('initted', 'update', 'step'))}
    else:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint.get('ema_model_state_dict', checkpoint)

    # Load state dict
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    print(f"Model loaded successfully on {device}")
    return model, vocab_char_map, device


def load_dataset(arrow_path):
    """Load the arrow dataset and keep only first entry per speaker"""
    print(f"Loading dataset from {arrow_path}...")

    # Load the arrow dataset
    dataset = Dataset_.from_file(arrow_path)

    print(f"Dataset loaded with {len(dataset)} samples")
    print(f"Dataset columns: {dataset.column_names}")

    # Extract text and speaker information
    all_texts = dataset['text']
    all_speakers = dataset['speaker'] if 'speaker' in dataset.column_names else ['unknown'] * len(dataset)

    return all_texts, all_speakers, set(all_speakers)


import torch

def extract_text_embeddings(model, vocab_char_map, device, batch_size=32):
    """
    Extract embeddings for every character in vocab_char_map from the model
    and return them as a matrix of shape (n_chars, embedding_dim), while
    printing embedding statistics.

    Args:
        model: The language model containing a text embedding layer.
        vocab_char_map: Dict mapping characters to integer IDs.
        device: Torch device ('cpu' or 'cuda').
        batch_size: Batch size for embedding extraction.

    Returns:
        embeddings_matrix: Tensor of shape (n_chars, embedding_dim)
                           where row i corresponds to the embedding of char i.
    """
    model.to(device)
    model.eval()

    char_ids = list(vocab_char_map.values())
    embeddings_list = []

    with torch.no_grad():
        for i in range(0, len(char_ids), batch_size):
            batch_ids = torch.tensor(char_ids[i:i+batch_size], device=device)
            batch_embeddings = model.transformer.language_module.text_embed(batch_ids)
            embeddings_list.append(batch_embeddings.cpu())

    embeddings_matrix = torch.cat(embeddings_list, dim=0)

    # Compute stats
    mean = embeddings_matrix.mean().item()
    std = embeddings_matrix.std().item()
    min_val = embeddings_matrix.min().item()
    max_val = embeddings_matrix.max().item()
    shape = embeddings_matrix.shape

    print(f"Embeddings stats:")
    print(f"  Shape: {shape}")
    print(f"  Mean: {mean:.4f}")
    print(f"  Std: {std:.4f}")
    print(f"  Min: {min_val:.4f}")
    print(f"  Max: {max_val:.4f}")

    return embeddings_matrix


def perform_dimensionality_reduction(embeddings, method='both'):
    """Perform dimensionality reduction using t-SNE and/or UMAP"""
    print(f"Performing dimensionality reduction...")

    n_samples = embeddings.shape[0]
    results = {}

    if method in ['tsne', 'both']:
        print("  Running t-SNE...")
        # Adjust perplexity based on number of samples (must be < n_samples)
        perplexity = min(30, max(5, n_samples - 1))
        print(f"  Using perplexity: {perplexity} (n_samples: {n_samples})")

        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=1000)
        tsne_result = tsne.fit_transform(embeddings)
        results['tsne'] = tsne_result
        print(f"  t-SNE completed. Shape: {tsne_result.shape}")

    if method in ['umap', 'both']:
        print("  Running UMAP...")
        # Adjust n_neighbors based on number of samples (must be < n_samples)
        n_neighbors = min(15, max(2, n_samples - 1))
        print(f"  Using n_neighbors: {n_neighbors} (n_samples: {n_samples})")

        umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors, min_dist=0.1)
        umap_result = umap_reducer.fit_transform(embeddings)
        results['umap'] = umap_result
        print(f"  UMAP completed. Shape: {umap_result.shape}")

    return results


def create_visualizations(embeddings_2d, speakers, texts, output_dir="text_embedding_visualizations"):
    """Create 2D visualizations colored by speaker"""
    print("Creating visualizations...")

    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    # Create color map for speakers
    unique_speakers = list(set(speakers))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_speakers)))
    speaker_color_map = {speaker: colors[i] for i, speaker in enumerate(unique_speakers)}

    for method, coords in embeddings_2d.items():
        # Create figure
        plt.figure(figsize=(15, 10))

        # Create scatter plot
        for speaker in unique_speakers:
            mask = np.array(speakers) == speaker
            if np.any(mask):
                plt.scatter(coords[mask, 0], coords[mask, 1],
                            c=[speaker_color_map[speaker]],
                            label=f"{speaker} ({np.sum(mask)} samples)",
                            alpha=0.7, s=20)

        plt.title(f'Text Embeddings Visualization - {method.upper()}', fontsize=16)
        plt.xlabel(f'{method.upper()} Component 1', fontsize=12)
        plt.ylabel(f'{method.upper()} Component 2', fontsize=12)

        # Add legend with speaker counts
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.tight_layout()

        # Save plot
        plt.savefig(f"{output_dir}/text_embeddings_{method}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{output_dir}/text_embeddings_{method}.pdf", bbox_inches='tight')
        print(f"  Saved {method} visualization")

        # Create a more detailed plot with speaker statistics
        plt.figure(figsize=(20, 12))

        # Main scatter plot
        plt.subplot(2, 2, (1, 2))
        for speaker in unique_speakers:
            mask = np.array(speakers) == speaker
            if np.any(mask):
                plt.scatter(coords[mask, 0], coords[mask, 1],
                            c=[speaker_color_map[speaker]],
                            label=f"{speaker}",
                            alpha=0.7, s=30)

        plt.title(f'Text Embeddings by Speaker - {method.upper()}', fontsize=16)
        plt.xlabel(f'{method.upper()} Component 1', fontsize=12)
        plt.ylabel(f'{method.upper()} Component 2', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

        # Speaker distribution histogram
        plt.subplot(2, 2, 3)
        speaker_counts = pd.Series(speakers).value_counts()
        speaker_counts.plot(kind='bar', color=[speaker_color_map[s] for s in speaker_counts.index])
        plt.title('Speaker Distribution', fontsize=14)
        plt.xlabel('Speaker', fontsize=12)
        plt.ylabel('Number of Samples', fontsize=12)
        plt.xticks(rotation=45)

        # Embedding variance by component
        plt.subplot(2, 2, 4)
        plt.scatter(coords[:, 0], coords[:, 1], c='gray', alpha=0.3, s=10)
        plt.title(f'Overall Distribution - {method.upper()}', fontsize=14)
        plt.xlabel(f'{method.upper()} Component 1', fontsize=12)
        plt.ylabel(f'{method.upper()} Component 2', fontsize=12)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/text_embeddings_{method}_detailed.png", dpi=300, bbox_inches='tight')
        print(f"  Saved detailed {method} visualization")

        plt.close('all')  # Close figures to free memory


def save_results(embeddings_original, embeddings_2d, speakers, texts, output_dir="text_embedding_visualizations"):
    """Save embedding data and analysis results"""
    print("Saving results...")

    Path(output_dir).mkdir(exist_ok=True)

    # Save original embeddings
    np.save(f"{output_dir}/original_embeddings.npy", embeddings_original)

    # Save 2D embeddings
    for method, coords in embeddings_2d.items():
        np.save(f"{output_dir}/embeddings_2d_{method}.npy", coords)

    # Create and save DataFrame with all information
    data = {
        'text': texts,
        'speaker': speakers,
    }

    # Add 2D coordinates
    for method, coords in embeddings_2d.items():
        data[f'{method}_x'] = coords[:, 0]
        data[f'{method}_y'] = coords[:, 1]

    df = pd.DataFrame(data)
    df.to_csv(f"{output_dir}/embeddings_data.csv", index=False)

    # Save summary statistics
    summary = {
        'total_samples': len(texts),
        'unique_speakers': len(set(speakers)),
        'speaker_counts': pd.Series(speakers).value_counts().to_dict(),
        'embedding_dim': embeddings_original.shape[1],
        'methods_used': list(embeddings_2d.keys())
    }

    with open(f"{output_dir}/summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Results saved to {output_dir}/")
    print(f"  - Original embeddings: original_embeddings.npy")
    print(f"  - 2D embeddings: embeddings_2d_*.npy")
    print(f"  - Data CSV: embeddings_data.csv")
    print(f"  - Summary: summary.json")


def main():
    """Main function to run the text embedding visualization"""
    print("=" * 60)
    print("Text Embedding Visualization for Romanian F5-TTS")
    print("=" * 60)

    # Configuration
    arrow_path = r"data\ro_tts_pinyin\raw_with_client_id.arrow"
    output_dir = "text_embedding_visualizations"
    batch_size = 16  # Adjust based on your GPU memory

    try:
        # Load model
        model, vocab_char_map, device = load_romanian_model()

        # Load dataset
        texts, speakers, unique_speakers = load_dataset(arrow_path)

        # Extract text embeddings
        embeddings = extract_text_embeddings(model, vocab_char_map, device)

        # Perform dimensionality reduction
        embeddings_2d = perform_dimensionality_reduction(embeddings, method='both')

        # Create visualizations
        create_visualizations(embeddings_2d, speakers, texts, output_dir)

        # Save results
        save_results(embeddings, embeddings_2d, speakers, texts, output_dir)

        print("\n" + "=" * 60)
        print("Text embedding visualization completed successfully!")
        print(f"Check the '{output_dir}' directory for results.")
        print("=" * 60)

    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
