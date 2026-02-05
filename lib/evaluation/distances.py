"""
Distance functions for embedding similarity computation.

Provides efficient pairwise distance calculations between embedding matrices,
commonly used in version identification and retrieval tasks.
"""

import torch


def pairwise_euclidean_distance_matrix(
    x: torch.Tensor, y: torch.Tensor, squared: bool = False, eps: float = 1e-6
) -> torch.Tensor:
    """
    Compute pairwise Euclidean distances between two sets of embeddings.

    Uses the efficient matrix formulation:
    ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x^T*y

    Args:
        x: First embedding matrix, shape (N, D)
        y: Second embedding matrix, shape (M, D)
        squared: If True, return squared distances. If False, return Euclidean distances
        eps: Small epsilon for numerical stability in sqrt operation

    Returns:
        Distance matrix of shape (N, M) where element (i,j) is the distance
        between x[i] and y[j]

    Example:
        >>> embeddings1 = torch.randn(100, 128)
        >>> embeddings2 = torch.randn(200, 128)
        >>> distances = pairwise_euclidean_distance_matrix(embeddings1, embeddings2)
        >>> distances.shape
        torch.Size([100, 200])
    """
    squared_x = x.pow(2).sum(1).view(-1, 1)
    squared_y = y.pow(2).sum(1).view(1, -1)
    dot_product = torch.mm(x, y.t())
    distance_matrix = squared_x - 2 * dot_product + squared_y
    # get rid of negative distances due to numerical instabilities
    distance_matrix[distance_matrix <= 0] = 0
    if not squared:
        # handle numerical stability
        # derivative of the square root operation applied to 0 is infinite
        # we need to handle by setting any 0 to eps
        mask = (distance_matrix == 0.0).type_as(distance_matrix)
        # use this mask to set indices with a value of 0 to eps
        distance_matrix += mask * eps
        # now it is safe to get the square root
        distance_matrix = torch.sqrt(distance_matrix)
        # undo the trick for numerical stability
        distance_matrix *= 1.0 - mask
    return distance_matrix


def pairwise_cosine_distance_matrix(
    x: torch.Tensor, y: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    Compute pairwise cosine distances between two sets of embeddings.

    Cosine distance is defined as: 1 - cosine_similarity
    where cosine_similarity = (x · y) / (||x|| * ||y||)

    Args:
        x: First embedding matrix, shape (N, D)
        y: Second embedding matrix, shape (M, D)
        eps: Small epsilon for numerical stability in normalization

    Returns:
        Distance matrix of shape (N, M) where element (i,j) is the cosine
        distance between x[i] and y[j]. Values range from 0 (identical) to 2
        (opposite directions)

    Example:
        >>> queries = torch.randn(50, 256)
        >>> candidates = torch.randn(1000, 256)
        >>> distances = pairwise_cosine_distance_matrix(queries, candidates)
        >>> distances.shape
        torch.Size([50, 1000])
        >>> # Find nearest neighbor for each query
        >>> nearest = distances.argmin(dim=1)
    """
    # Normalize x and y to unit vectors (L2 normalization)
    x_norm = x / (x.norm(dim=1, keepdim=True) + eps)
    y_norm = y / (y.norm(dim=1, keepdim=True) + eps)

    # Compute cosine similarity
    cosine_similarity = torch.mm(x_norm, y_norm.t())

    # Convert to cosine distance
    cosine_distance = 1 - cosine_similarity

    return cosine_distance
