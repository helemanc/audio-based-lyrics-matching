import torch
def pairwise_euclidean_distance_matrix(x, y, squared=False, eps=1e-6):
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


def pairwise_cosine_distance_matrix(x, y, eps=1e-6):
    # Normalize x and y to unit vectors (L2 normalization)
    x_norm = x / (x.norm(dim=1, keepdim=True) + eps)
    y_norm = y / (y.norm(dim=1, keepdim=True) + eps)

    # Compute cosine similarity
    cosine_similarity = torch.mm(x_norm, y_norm.t())

    # Convert to cosine distance
    cosine_distance = 1 - cosine_similarity

    return cosine_distance