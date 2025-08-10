import torch
import torch.nn as nn
import torch.nn.functional as F


class VQEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)

    def forward(self, z):
        b, c, h, w = z.shape
        assert c == self.embedding_dim, f"Input channel {c} does not match embedding dim {self.embedding_dim}"
        z_channel_last = z.permute(0, 2, 3, 1)
        z_flattened = z_channel_last.reshape(b * h * w, self.embedding_dim)

        # Calculate distances between z and the codebook embeddings |a-b|²
        distances = (
            torch.sum(z_flattened**2, dim=-1, keepdim=True)  # a²
            + torch.sum(self.embedding.weight.t() ** 2, dim=0, keepdim=True)  # b²
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())  # -2ab
        )

        # Get the index with the smallest distance
        encoding_indices = torch.argmin(distances, dim=-1)

        # Get the quantized vector
        z_q = self.embedding(encoding_indices)
        z_q = z_q.reshape(b, h, w, self.embedding_dim)
        z_q = z_q.permute(0, 3, 1, 2)

        # Calculate the commitment loss
        loss = F.mse_loss(z_q, z.detach()) + self.commitment_cost * F.mse_loss(
            z_q.detach(), z
        )

        # Straight-through estimator trick for gradient backpropagation
        z_q = z + (z_q - z).detach()

        return z_q, loss, encoding_indices


if __name__ == "__main__":
    # --- Configuration ---
    num_embeddings = 512
    embedding_dim = 64
    commitment_cost = 0.25
    batch_size = 4
    height = 16
    width = 16

    # --- Test ---
    # Instantiate the module
    vq_embedding = VQEmbedding(num_embeddings, embedding_dim, commitment_cost)
    print("VQEmbedding module instantiated:")
    print(vq_embedding)

    # Create a random input tensor
    input_tensor = torch.randn(batch_size, embedding_dim, height, width)
    print(f"\nInput tensor shape: {input_tensor.shape}")

    # Forward pass
    try:
        quantized_tensor, loss, indices = vq_embedding(input_tensor)

        # --- Verification ---
        print("\n--- Results ---")
        print(f"Quantized tensor shape: {quantized_tensor.shape}")
        print(f"Indices shape: {indices.shape}")
        print(f"Loss: {loss.item():.4f}")

        # Check if shapes are correct
        assert input_tensor.shape == quantized_tensor.shape
        assert indices.shape == (batch_size * height * width,)
        print("\nShape assertions passed!")

        # Check wrong dimension
        print("\n--- Testing Error Handling ---")
        wrong_dim_tensor = torch.randn(batch_size, embedding_dim + 1, height, width)
        try:
            vq_embedding(wrong_dim_tensor)
        except AssertionError as e:
            print(f"Successfully caught error for wrong input dimension: {e}")

    except Exception as e:
        print(f"\nAn error occurred during the test: {e}")