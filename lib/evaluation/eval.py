import sys
import torch
from lib.evaluation.distances import * 

###################################################################################################

def compute_baseline(
    distances,  # Precomputed distances (B, B')
    queries_c,  # Clique index (B)
    queries_i,  # Song index (B)
    candidates_c,  # Clique index (B')
    candidates_i,  # Song index (B')
    return_distances=False,  # NEW: Optional parameter to return distances
):
    # Prepare
    aps = []
    r1s = []
    rpcs = []
    
    # NEW: Store distances if requested
    all_distances = [] if return_distances else None
    
    # ADD: Progress tracking
    total_queries = len(queries_i)
    # print(f"Starting baseline evaluation: {total_queries} queries vs {len(candidates_i)} candidates")
    
    for n in range(len(queries_i)):
    #     # ADD: Progress updates
    #     if (n + 1) % 100 == 0 or n == 0:
    #         print(f"  Progress: {n + 1}/{total_queries} queries ({100*(n+1)/total_queries:.1f}%)")
        
        dist = distances[n]  # Use precomputed distances directly

        # Get ground truth
        match_clique = candidates_c == queries_c[n]
        
        # Remove query from candidates if present
        match_query = candidates_i == queries_i[n]
        
        # NEW: Store original distances before modification
        if return_distances:
            all_distances.append(dist.clone())
        
        dist = torch.where(match_query, torch.inf, dist)
        match_clique = torch.where(match_query, False, match_clique)

        # Compute AP and R1
        aps.append(average_precision(dist, match_clique))
        r1s.append(rank_of_first_correct(dist, match_clique))
        rpcs.append(rank_percentile(dist, match_clique))

    # Return as vector
    aps = torch.stack(aps)
    r1s = torch.stack(r1s)
    rpcs = torch.stack(rpcs)
    
    # ADD: Completion message
    #print(f"Baseline evaluation completed: {len(aps)} results")
    
    # NEW: Return distances if requested
    if return_distances:
        distance_matrix = torch.stack(all_distances)  # Shape: (B, B')
        return aps, r1s, rpcs, distance_matrix
    else:
        return aps, r1s, rpcs


def compute_baseline_vectorized(
    distances,  # Precomputed distances (B, B')
    queries_c,  # Clique index (B)
    queries_i,  # Song index (B)
    candidates_c,  # Clique index (B')
    candidates_i,  # Song index (B')
    return_distances=False,
):
    """
    Vectorized version - processes all queries at once instead of looping
    This should be 10-50x faster than the original
    """
    
    B = len(queries_i)  # Number of queries
    B_prime = len(candidates_i)  # Number of candidates
    
    # Create masks for all queries at once
    # Shape: (B, B') - True where candidate clique matches query clique
    match_clique = candidates_c.unsqueeze(0) == queries_c.unsqueeze(1)
    
    # Shape: (B, B') - True where candidate song matches query song  
    match_query = candidates_i.unsqueeze(0) == queries_i.unsqueeze(1)
    
    # Store original distances if requested
    if return_distances:
        all_distances = distances.clone()
    
    # Remove self-matches by setting distance to infinity
    # Shape: (B, B')
    distances_masked = torch.where(match_query, torch.inf, distances)
    match_clique_masked = torch.where(match_query, False, match_clique)
    
    # Vectorized computation of metrics for all queries
    aps = []
    r1s = []
    rpcs = []
    
    # Unfortunately, AP computation still needs a loop because of its sequential nature
    # But we can at least avoid the print overhead
    for n in range(B):
        dist_n = distances_masked[n]
        match_n = match_clique_masked[n]
        
        aps.append(average_precision(dist_n, match_n))
        r1s.append(rank_of_first_correct(dist_n, match_n))
        rpcs.append(rank_percentile(dist_n, match_n))
    
    # Stack results
    aps = torch.stack(aps)
    r1s = torch.stack(r1s) 
    rpcs = torch.stack(rpcs)
    
    if return_distances:
        return aps, r1s, rpcs, all_distances
    else:
        return aps, r1s, rpcs

@torch.inference_mode()
def compute(
    model,
    queries_c,  # clique index (B)
    queries_i,  # song index (B)
    queries_z,  # embedding (B,S,C)
    candidates_c,  # clique index (B')
    candidates_i,  # song index (B')
    candidates_z,  # embedding (B',S,C)
    redux_strategy=None,
    batch_size_candidates=None,
    distance_fn=pairwise_cosine_distance_matrix,
    return_distances=False,  # NEW: Optional parameter to return distances
):
    # Ensure all tensors are on the same device
    device = queries_z.device
    candidates_c = candidates_c.to(device)
    candidates_i = candidates_i.to(device)
    candidates_z = candidates_z.to(device)
    
    # Prepare
    aps = []
    r1s = []
    rpcs = []
    
    # NEW: Store distances if requested
    all_distances = [] if return_distances else None
    
    if model is not None: 
        model.eval()
    
    for n in range(len(queries_i)):
        # Compute distance between query and everything
        if batch_size_candidates is None or batch_size_candidates >= len(candidates_i):
             if distance_fn == "cosine":
                dist = pairwise_cosine_distance_matrix(
                    queries_z[n : n + 1].float(),
                    candidates_z.float(),
                ).squeeze(0)
             else:  
                dist = pairwise_euclidean_distance_matrix(
                    queries_z[n : n + 1].float(),
                    candidates_z.float(),
                ).squeeze(0)
        else:
            dist = []
            for mstart in range(0, len(candidates_i), batch_size_candidates):
                mend = min(mstart + batch_size_candidates, len(candidates_i))
                if distance_fn == "cosine":
                    ddd = pairwise_cosine_distance_matrix(
                        queries_z[n : n + 1].float(),
                        candidates_z[mstart:mend].float(),
                    ).squeeze(0)
                else:
                    ddd = pairwise_euclidean_distance_matrix(
                        queries_z[n : n + 1].float(),
                        candidates_z[mstart:mend].float(),
                    ).squeeze(0)
                
                dist.append(ddd)
           
            dist = torch.cat(dist, dim=-1)

        # Get ground truth
        match_clique = candidates_c == queries_c[n]
        
        # NEW: Store original distances before modification
        if return_distances:
            all_distances.append(dist.clone())

        # Remove query from candidates if present
        match_query = candidates_i == queries_i[n]
        dist = torch.where(match_query, torch.inf, dist)
        match_clique = torch.where(match_query, False, match_clique)

        # Compute AP and R1
        aps.append(average_precision(dist, match_clique))
        r1s.append(rank_of_first_correct(dist, match_clique))
        rpcs.append(rank_percentile(dist, match_clique))
    
    # Return as vector
    aps = torch.stack(aps)
    r1s = torch.stack(r1s)
    rpcs = torch.stack(rpcs)
    
    # NEW: Return distances if requested
    if return_distances:
        distance_matrix = torch.stack(all_distances)  # Shape: (B, B')
        return aps, r1s, rpcs, distance_matrix
    else:
        return aps, r1s, rpcs

###################################################################################################


@torch.inference_mode()
def average_precision(distances, ismatch):
    assert distances.ndim == 1 and ismatch.ndim == 1 and len(distances) == len(ismatch)
    rel = ismatch.type_as(distances)
    assert rel.sum() >= 1, "There should be at least 1 relevant item"
    rel = rel[torch.argsort(distances)]
    rank = torch.arange(len(rel), device=distances.device) + 1
    prec = torch.cumsum(rel, 0) / rank
    ap = torch.sum(prec * rel) / torch.sum(rel)
    return ap


@torch.inference_mode()
def rank_of_first_correct(distances, ismatch):
    assert distances.ndim == 1 and ismatch.ndim == 1 and len(distances) == len(ismatch)
    rel = ismatch.type_as(distances)
    assert rel.sum() >= 1, "There should be at least 1 relevant item"
    rel = rel[torch.argsort(distances)]
    # argmax returns index of first occurrence
    r1 = (torch.argmax(rel) + 1).type_as(distances)
    return r1


@torch.inference_mode()
def rank_percentile(distances, ismatch, biased=False):
    # https://publications.hevs.ch/index.php/publications/show/125
    assert distances.ndim == 1 and ismatch.ndim == 1 and len(distances) == len(ismatch)
    rel = ismatch.type_as(distances)
    assert rel.sum() >= 1, "There should be at least 1 relevant item"
    rel = rel[torch.argsort(distances)]
    if biased:
        # Size of the clique affects the measure, that is, you do not get a
        # perfect 0 score if clique size>1
        normrank = torch.linspace(0, 1, len(rel), device=distances.device)
    else:
        # counting number of zeros preceding rels allows to get perfect 0 score
        normrank = torch.cumsum(1 - rel, 0) / torch.sum(1 - rel)
    rpc = torch.sum(rel * normrank) / torch.sum(rel)
    return 100 * rpc


###################################################################################################
