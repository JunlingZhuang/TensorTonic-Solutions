import torch
import torch.nn as nn

def create_embedding_layer(vocab_size: int, d_model: int) -> nn.Embedding:
    """
    Returns an embedding layer with the requested dimensions.
    """
    layer = nn.Embedding(num_embeddings = vocab_size, embedding_dim = d_model)
    return layer

def embed_tokens(embedding: nn.Embedding, tokens: torch.Tensor, d_model: int) -> torch.Tensor:
    """
    Returns scaled token embeddings.
    """
    vectors = embedding(tokens)
    scale = math.sqrt(d_model)
    return vectors * scale
    

