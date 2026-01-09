import os 
from tqdm import tqdm
from lib.evaluation.eval import compute
import torch 
import math 
import json 
from pathlib import Path


def evaluate_latents(query_z, cand_z, latent_type, distances, query_c, query_i, 
                     cand_c, cand_i, fabric):
    """
    Evaluate latents for given queries and candidates.
    
    Args:
        query_z: Query latents (encoder or decoder)
        cand_z: Candidate latents (encoder or decoder)
        latent_type: String identifier ('encoder' or 'decoder')
        distances: List of distance functions to evaluate
        query_c, query_i, cand_c, cand_i: Additional query/candidate data
        fabric: Fabric instance for distributed computing
    """
    results = {}
    
    for dist in distances:
        # Evaluate each query
        aps = []
        r1s = []
        rpcs = []
        
        for n in tqdm(range(len(query_z)), desc="Retrieve", disable=not fabric.is_global_zero):
            ap, r1, rpc = compute(
                model=None,
                queries_c=query_c[n:n+1],
                queries_i=query_i[n:n+1],
                queries_z=query_z[n:n+1],
                candidates_c=cand_c,
                candidates_i=cand_i,
                candidates_z=cand_z,
                redux_strategy=None,
                batch_size_candidates=2**15,
                distance_fn=dist
            )
            aps.append(ap)
            r1s.append(r1)
            rpcs.append(rpc)

        # Stack metrics
        aps = torch.stack(aps)
        r1s = torch.stack(r1s)
        rpcs = torch.stack(rpcs)

        # Collect metrics from all GPUs
        fabric.barrier()
        aps = fabric.all_gather(aps)
        r1s = fabric.all_gather(r1s)
        rpcs = fabric.all_gather(rpcs)
        
        # Flatten the gathered metrics
        aps = torch.cat(torch.unbind(aps, dim=0), dim=0)
        r1s = torch.cat(torch.unbind(r1s, dim=0), dim=0)
        rpcs = torch.cat(torch.unbind(rpcs, dim=0), dim=0)

        if fabric.is_global_zero:
            logdict_mean = {
                "MAP": aps.mean().item(),
                "MR1": r1s.mean().item(),
                "ARP": rpcs.mean().item(),
            }
            
            logdict_ci = {
                "MAP": 1.96 * aps.std().item() / math.sqrt(len(aps)),
                "MR1": 1.96 * r1s.std().item() / math.sqrt(len(r1s)),
                "ARP": 1.96 * rpcs.std().item() / math.sqrt(len(rpcs)),
            }
            
            print("=" * 100)
            print(f"Result {latent_type.capitalize()} Latents - Distance {dist}:")
            print(f"  Avg --> MAP: {logdict_mean['MAP']:.4f}, MR1: {logdict_mean['MR1']:.4f}, ARP: {logdict_mean['ARP']:.4f}")
            print(f"  c.i. -> MAP: {logdict_ci['MAP']:.4f}, MR1: {logdict_ci['MR1']:.4f}, ARP: {logdict_ci['ARP']:.4f}")
            print("=" * 100)

            results[f"{latent_type}_latents_{dist}"] = {
                "mean": logdict_mean,
                f"confidence_interval_{dist}": logdict_ci
            }
    
    return results


# Main evaluation code
def run_evaluation(query_z_e, query_z_d, cand_z_e, cand_z_d, distances, 
                   query_c, query_i, cand_c, cand_i, fabric, dataset_name, decoding_config_name):
    """Run evaluation for both encoder and decoder latents."""
    
    # Evaluate encoder latents
    encoder_results = evaluate_latents(
        query_z_e, cand_z_e, 'encoder', distances,
        query_c, query_i, cand_c, cand_i, fabric
    )
    
    # Evaluate decoder latents
    decoder_results = evaluate_latents(
        query_z_d, cand_z_d, 'decoder', distances,
        query_c, query_i, cand_c, cand_i, fabric
    )
    
    # Combine results and save
    if fabric.is_global_zero:
        all_results = {**encoder_results, **decoder_results}
        
        results_file = f"evaluation_results_{decoding_config_name}.json"
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=4)
        
        print(f"Results saved to {results_file}")