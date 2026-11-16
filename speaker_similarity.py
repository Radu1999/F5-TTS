"""
Speaker Similarity Module using TitaNet-L Model

This module provides functionality to compute speaker similarity scores between
reference and generated audio using NVIDIA NeMo's TitaNet-L speaker verification model.
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path
import logging
from typing import Optional, Tuple
import tempfile
import os

try:
    import nemo.collections.asr as nemo_asr
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False
    logging.warning("NVIDIA NeMo not available. Speaker similarity scoring will be disabled.")


class SpeakerSimilarityScorer:
    """Speaker similarity scorer using TitaNet-L model"""
    
    def __init__(self, device: str = "auto"):
        """
        Initialize the speaker similarity scorer
        
        Args:
            device: Device to use ("cpu", "cuda", or "auto")
        """
        self.device = self._get_device(device)
        self.model = None
        self.target_sample_rate = 16000  # TitaNet-L expects 16kHz audio
        
        if not NEMO_AVAILABLE:
            raise ImportError(
                "NVIDIA NeMo is required for speaker similarity scoring. "
                "Install with: pip install nemo_toolkit[asr]"
            )
    
    def _get_device(self, device: str) -> str:
        """Determine the appropriate device"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def load_model(self) -> None:
        """Load the TitaNet-L model"""
        if self.model is not None:
            return
        
        try:
            print("Loading TitaNet-L speaker verification model...")
            self.model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
                model_name='titanet_large'
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            print(f"TitaNet-L model loaded successfully on {self.device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load TitaNet-L model: {e}")
    
    def _preprocess_audio(self, audio_path: str) -> str:
        """
        Preprocess audio to match TitaNet-L requirements (16kHz, mono)
        
        Args:
            audio_path: Path to input audio file
            
        Returns:
            Path to preprocessed audio file (temporary file if conversion needed)
        """
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample to 16kHz if needed
            if sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=self.target_sample_rate
                )
                waveform = resampler(waveform)
            
            # If no conversion needed, return original path
            if sample_rate == self.target_sample_rate and waveform.shape[0] == 1:
                return audio_path
            
            # Save to temporary file
            temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
            os.close(temp_fd)  # Close file descriptor, we'll use torchaudio to save
            
            torchaudio.save(temp_path, waveform, self.target_sample_rate)
            return temp_path
            
        except Exception as e:
            raise RuntimeError(f"Failed to preprocess audio {audio_path}: {e}")
    
    def get_embedding(self, audio_path: str) -> torch.Tensor:
        """
        Extract speaker embedding from audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Speaker embedding tensor
        """
        if self.model is None:
            self.load_model()
        
        # Preprocess audio
        processed_path = self._preprocess_audio(audio_path)
        temp_file_created = processed_path != audio_path
        
        try:
            # Extract embedding
            with torch.no_grad():
                embedding = self.model.get_embedding(processed_path).squeeze()
            
            # Normalize embedding
            embedding = embedding / torch.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            raise RuntimeError(f"Failed to extract embedding from {audio_path}: {e}")
        
        finally:
            # Clean up temporary file if created
            if temp_file_created and os.path.exists(processed_path):
                try:
                    os.unlink(processed_path)
                except OSError:
                    pass  # Ignore cleanup errors
    
    def compute_similarity(self, ref_audio_path: str, gen_audio_path: str) -> float:
        """
        Compute speaker similarity score between reference and generated audio
        
        Args:
            ref_audio_path: Path to reference audio file
            gen_audio_path: Path to generated audio file
            
        Returns:
            Similarity score between 0 and 1 (1 = identical speaker)
        """
        try:
            # Extract embeddings
            ref_embedding = self.get_embedding(ref_audio_path)
            gen_embedding = self.get_embedding(gen_audio_path)
            
            # Compute cosine similarity
            similarity = torch.dot(ref_embedding, gen_embedding).item()
            
            # Convert from [-1, 1] to [0, 1] range
            similarity_score = (similarity + 1) / 2
            
            return similarity_score
            
        except Exception as e:
            logging.error(f"Failed to compute similarity between {ref_audio_path} and {gen_audio_path}: {e}")
            return 0.0  # Return 0 similarity on error
    
    def compute_similarity_batch(self, audio_pairs: list) -> list:
        """
        Compute similarity scores for a batch of audio pairs
        
        Args:
            audio_pairs: List of (reference_path, generated_path) tuples
            
        Returns:
            List of similarity scores
        """
        similarities = []
        
        for ref_path, gen_path in audio_pairs:
            similarity = self.compute_similarity(ref_path, gen_path)
            similarities.append(similarity)
        
        return similarities


def create_speaker_similarity_scorer(device: str = "auto") -> Optional[SpeakerSimilarityScorer]:
    """
    Factory function to create a speaker similarity scorer
    
    Args:
        device: Device to use ("cpu", "cuda", or "auto")
        
    Returns:
        SpeakerSimilarityScorer instance or None if NeMo is not available
    """
    if not NEMO_AVAILABLE:
        logging.warning("NVIDIA NeMo not available. Speaker similarity scoring disabled.")
        return None
    
    try:
        scorer = SpeakerSimilarityScorer(device=device)
        return scorer
    except Exception as e:
        logging.error(f"Failed to create speaker similarity scorer: {e}")
        return None


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python speaker_similarity.py <reference_audio> <generated_audio>")
        sys.exit(1)
    
    ref_audio = sys.argv[1]
    gen_audio = sys.argv[2]
    
    # Create scorer
    scorer = create_speaker_similarity_scorer()
    if scorer is None:
        print("Failed to create speaker similarity scorer")
        sys.exit(1)
    
    # Compute similarity
    try:
        similarity = scorer.compute_similarity(ref_audio, gen_audio)
        print(f"Speaker similarity score: {similarity:.4f} ({similarity*100:.2f}%)")
    except Exception as e:
        print(f"Error computing similarity: {e}")
        sys.exit(1)
