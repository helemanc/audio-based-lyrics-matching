"""
Baseline evaluation methods for version identification.

This module provides implementations of transcription-based baselines for
version identification, including:
- SBERT (Sentence-BERT) embeddings
- TF-IDF with cosine similarity
- TF-IDF with Lucene-style "More Like This"
- Theoretical bounds (Ideal, Random, Modified variants)

All methods work with Whisper transcriptions treated as lyrics for
audio-based version identification tasks.
"""

import gc
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from lib.evaluation.eval import compute_baseline


class TFIDFEvaluator:
    """
    TF-IDF implementation for version identification using transcriptions.

    Supports two similarity approaches:
    1. Cosine similarity: Standard TF-IDF vectors with cosine distance
    2. Lucene-style: Elasticsearch "More Like This" query simulation

    Both approaches support top-k filtering to reduce candidate space,
    as described in the paper.

    Args:
        max_query_terms: Maximum terms to select for Lucene-style queries
        min_term_freq: Minimum term frequency threshold
        min_doc_freq: Minimum document frequency threshold

    Example:
        >>> evaluator = TFIDFEvaluator(max_query_terms=12)
        >>> results = evaluator.compute_tfidf_baseline_cover_song_detection(
        ...     transcriptions=['hello world', 'hello there'],
        ...     clique_ids=torch.tensor([0, 0]),
        ...     version_ids=torch.tensor([0, 1]),
        ...     method='cosine'
        ... )
    """

    def __init__(
        self, max_query_terms: int = 12, min_term_freq: int = 1, min_doc_freq: int = 2
    ):
        # TF-IDF vectorizer configured like the paper's lyrics approach
        self.tfidf_vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),  # unigrams and bigrams as in paper
            max_features=10000,  # reasonable limit for transcriptions
            min_df=min_doc_freq,
            max_df=0.95,  # ignore terms in more than 95% of documents
            strip_accents="unicode",
            token_pattern=r"\b[a-zA-Z][a-zA-Z]+\b",  # alphabetic tokens 2+ chars
        )

        # Lucene-style parameters
        self.max_query_terms = max_query_terms
        self.min_term_freq = min_term_freq
        self.min_doc_freq = min_doc_freq

        # Stored corpus data for Lucene-style approach
        self.corpus = None
        self.tfidf_matrix = None
        self.feature_names = None
        self.idf_values = None

    def preprocess_transcription_as_lyrics(self, transcription: str) -> str:
        """
        Preprocess transcriptions as lyrics following the paper's bag-of-words approach
        """
        if not transcription or not isinstance(transcription, str):
            return ""

        # Convert to lowercase
        text = transcription.lower()

        # Remove common transcription artifacts while preserving lyrical content
        # Remove timestamps, speaker markers, etc.
        text = re.sub(r"\[\d+:\d+\]", "", text)  # Remove [mm:ss] timestamps
        text = re.sub(r"\(.*?\)", "", text)  # Remove parenthetical annotations
        text = re.sub(r"\[.*?\]", "", text)  # Remove bracketed annotations

        # Remove excessive filler words that don't contribute to song content
        # But keep words that might actually be lyrics
        excessive_fillers = r"\b(um|uh|ah|hmm|er|eh)\b"
        text = re.sub(excessive_fillers, " ", text)

        # Clean up punctuation but preserve apostrophes in contractions
        text = re.sub(r"[^\w\s']", " ", text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def fit_corpus(self, transcriptions: List[str]):
        """
        Fit the TF-IDF model on the corpus (needed for Lucene-style approach)
        """
        print("Building TF-IDF corpus...")

        # Preprocess all transcriptions
        self.corpus = [
            self.preprocess_transcription_as_lyrics(text) for text in transcriptions
        ]

        # Filter out empty documents for fitting
        non_empty_corpus = [text for text in self.corpus if text.strip()]

        if not non_empty_corpus:
            raise ValueError("No valid transcriptions found!")

        # Fit TF-IDF vectorizer on all documents (including empty ones)
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.corpus)
        self.feature_names = self.tfidf_vectorizer.get_feature_names_out()
        self.idf_values = self.tfidf_vectorizer.idf_
        self.vocab_to_idx = self.tfidf_vectorizer.vocabulary_

        print(f"Corpus size: {len(self.corpus)}")
        print(f"Vocabulary size: {len(self.feature_names)}")
        print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")

    def extract_top_tfidf_terms(self, query_idx: int) -> List[Tuple[str, float]]:
        """
        Extract top-k terms with highest TF-IDF scores from query document
        This implements the "More Like This" term selection from the paper
        """
        if query_idx >= self.tfidf_matrix.shape[0]:
            return []

        # Get TF-IDF vector for query document
        query_vector = self.tfidf_matrix[query_idx]

        # Get non-zero terms and their scores
        _, term_indices = query_vector.nonzero()
        term_scores = query_vector.data

        # Create list of (term, tfidf_score) tuples
        term_score_pairs = [
            (self.feature_names[idx], score)
            for idx, score in zip(term_indices, term_scores)
        ]

        # Sort by TF-IDF score (descending) and take top terms
        # This is exactly what Elasticsearch "More Like This" does
        term_score_pairs.sort(key=lambda x: x[1], reverse=True)

        return term_score_pairs[: self.max_query_terms]

    def create_query_vector_from_terms(
        self, selected_terms: List[Tuple[str, float]], vocab_size: int
    ) -> np.ndarray:
        """
        Create a query vector using only the selected terms
        This simulates the disjunctive query formation in Elasticsearch
        """
        query_vector = np.zeros(vocab_size)

        for term, score in selected_terms:
            if term in self.tfidf_vectorizer.vocabulary_:
                term_idx = self.tfidf_vectorizer.vocabulary_[term]
                query_vector[term_idx] = score

        return query_vector

    def more_like_this_similarity(self, query_idx: int) -> np.ndarray:
        """
        Compute "More Like This" similarity as described in the paper:
        1. Select top-k terms with highest TF-IDF from query
        2. Form query vector with these terms
        3. Compute cosine similarity with all documents
        """
        # Step 1: Extract top TF-IDF terms from query
        selected_terms = self.extract_top_tfidf_terms(query_idx)

        if not selected_terms:
            return np.zeros(self.tfidf_matrix.shape[0])

        # Step 2: Create query vector using only selected terms
        vocab_size = len(self.tfidf_vectorizer.vocabulary_)
        query_vector = self.create_query_vector_from_terms(selected_terms, vocab_size)

        # Step 3: Compute cosine similarity with all documents
        # Reshape query vector to 2D for cosine_similarity
        query_vector_2d = query_vector.reshape(1, -1)

        # Compute similarities
        similarities = cosine_similarity(query_vector_2d, self.tfidf_matrix).flatten()

        return similarities

    def compute_cover_song_similarities_cosine(
        self, transcriptions: List[str], top_k: int = 100, compute_all: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Compute cover song similarities using simple cosine similarity approach

        Args:
            transcriptions: List of transcription texts
            top_k: Top-k filtering parameter
            compute_all: Whether to also compute similarities without top-k filtering

        Returns:
            Dictionary with 'top_k' and optionally 'all' similarity matrices
        """
        print(
            f"Computing cover song similarities (COSINE) for {len(transcriptions)} transcriptions..."
        )

        # Preprocess transcriptions as lyrics
        print("Preprocessing transcriptions as lyrics...")
        processed_texts = []
        valid_mask = []

        for text in tqdm(transcriptions, desc="Preprocessing"):
            if text is None or (isinstance(text, str) and not text.strip()):
                processed_texts.append("")
                valid_mask.append(False)
            else:
                processed_text = self.preprocess_transcription_as_lyrics(str(text))
                processed_texts.append(processed_text)
                valid_mask.append(bool(processed_text.strip()))

        valid_mask = torch.tensor(valid_mask, dtype=torch.bool)
        print(f"Valid transcriptions: {valid_mask.sum().item()}/{len(transcriptions)}")

        # Handle case where no valid transcriptions exist
        if valid_mask.sum() == 0:
            print("Warning: No valid transcriptions found!")
            n = len(transcriptions)
            result = {"top_k": torch.ones((n, n), dtype=torch.float32)}
            if compute_all:
                result["all"] = torch.ones((n, n), dtype=torch.float32)
            return result

        # Fit TF-IDF vectorizer on all non-empty texts
        print("Fitting TF-IDF vectorizer...")
        non_empty_texts = [text for text in processed_texts if text.strip()]

        try:
            self.tfidf_vectorizer.fit(non_empty_texts)
            print(f"TF-IDF vocabulary size: {len(self.tfidf_vectorizer.vocabulary_)}")
        except ValueError as e:
            print(f"TF-IDF fitting failed: {e}")
            n = len(transcriptions)
            result = {"top_k": torch.ones((n, n), dtype=torch.float32)}
            if compute_all:
                result["all"] = torch.ones((n, n), dtype=torch.float32)
            return result

        # Transform all texts to TF-IDF vectors
        print("Transforming texts to TF-IDF vectors...")
        try:
            tfidf_matrix = self.tfidf_vectorizer.transform(processed_texts)
            print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        except Exception as e:
            print(f"TF-IDF transformation failed: {e}")
            n = len(transcriptions)
            result = {"top_k": torch.ones((n, n), dtype=torch.float32)}
            if compute_all:
                result["all"] = torch.ones((n, n), dtype=torch.float32)
            return result

        # Initialize similarity matrices
        n = tfidf_matrix.shape[0]
        similarities_all = torch.zeros((n, n), dtype=torch.float32)

        # Query-based approach: for each song, find similar covers using cosine similarity
        print("Computing cosine similarities...")

        for query_idx in tqdm(range(n), desc="Processing queries"):
            if not valid_mask[query_idx]:
                # If query transcription is invalid, no similarities
                similarities_all[query_idx, :] = 0.0
                continue

            # Get query vector
            query_vector = tfidf_matrix[query_idx : query_idx + 1]

            # Compute similarities with all candidates
            candidate_similarities = cosine_similarity(
                query_vector, tfidf_matrix
            ).flatten()

            # Convert to torch tensor
            similarities_all[query_idx, :] = torch.from_numpy(
                candidate_similarities
            ).float()

            # Set self-similarity to 0 (don't retrieve the query itself)
            similarities_all[query_idx, query_idx] = 0.0

        # Prepare results
        result = {}

        # Store all similarities if requested
        if compute_all:
            result["all"] = similarities_all.clone()

        # Apply top-k filtering for top_k results
        similarities_top_k = similarities_all.clone()
        if top_k < n:
            print(f"Applying top-{top_k} filtering...")
            for query_idx in range(n):
                if valid_mask[query_idx]:
                    # Get top-k most similar candidates
                    _, top_indices = torch.topk(
                        similarities_top_k[query_idx, :], k=min(top_k, n - 1)
                    )

                    # Create mask for top-k
                    top_k_mask = torch.zeros(n, dtype=torch.bool)
                    top_k_mask[top_indices] = True

                    # Zero out similarities for candidates not in top-k
                    similarities_top_k[query_idx, ~top_k_mask] = 0.0

        result["top_k"] = similarities_top_k

        return result

    def compute_cover_song_similarities_lucene_correct(
        self, transcriptions: List[str], top_k: int = 100, compute_all: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Compute similarities using the paper's correct methodology

        Args:
            transcriptions: List of transcription texts
            top_k: Top-k filtering parameter
            compute_all: Whether to also compute similarities without top-k filtering

        Returns:
            Dictionary with 'top_k' and optionally 'all' similarity matrices
        """
        print(
            f"Computing paper-correct Lucene similarities for {len(transcriptions)} transcriptions..."
        )

        # Fit corpus
        self.fit_corpus(transcriptions)

        n = len(transcriptions)
        similarities_all = torch.zeros((n, n), dtype=torch.float32)

        # Check which documents are valid (non-empty)
        valid_mask = torch.tensor(
            [self.tfidf_matrix[i].nnz > 0 for i in range(n)], dtype=torch.bool
        )

        print(f"Valid documents: {valid_mask.sum().item()}/{n}")

        # Process each query using "More Like This" approach
        for query_idx in tqdm(range(n), desc="Processing queries"):
            if not valid_mask[query_idx]:
                continue

            # Compute similarities using More Like This
            query_similarities = self.more_like_this_similarity(query_idx)

            # Convert to torch tensor
            similarities_all[query_idx, :] = torch.from_numpy(
                query_similarities
            ).float()

            # Don't retrieve the query itself
            similarities_all[query_idx, query_idx] = 0.0

        # Prepare results
        result = {}

        # Store all similarities if requested
        if compute_all:
            result["all"] = similarities_all.clone()

        # Apply top-k filtering for top_k results
        similarities_top_k = similarities_all.clone()
        if top_k < n:
            print(f"Applying top-{top_k} filtering...")
            for query_idx in range(n):
                if valid_mask[query_idx]:
                    _, top_indices = torch.topk(
                        similarities_top_k[query_idx, :], k=min(top_k, n - 1)
                    )
                    top_k_mask = torch.zeros(n, dtype=torch.bool)
                    top_k_mask[top_indices] = True
                    similarities_top_k[query_idx, ~top_k_mask] = 0.0

        result["top_k"] = similarities_top_k

        return result

    def compute_tfidf_baseline_cover_song_detection(
        self,
        transcriptions: List[str],
        clique_ids: torch.Tensor,
        version_ids: torch.Tensor,
        method: str = "cosine",
        top_k: int = 100,
        compute_all: bool = True,
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Compute TF-IDF baseline for cover song detection using transcriptions as lyrics

        Args:
            transcriptions: List of transcription texts
            clique_ids: Clique IDs for evaluation
            version_ids: Version IDs for evaluation
            method: 'cosine' or 'lucene' for different similarity approaches
            top_k: Top-k filtering parameter
            compute_all: Whether to also compute without top-k filtering

        Returns:
            Dictionary with evaluation metrics for 'top_k' and optionally 'all' approaches
        """
        print(
            f"Computing TF-IDF baseline for cover song detection (method: {method.upper()})..."
        )

        # Compute similarities based on method
        if method.lower() == "cosine":
            similarities_dict = self.compute_cover_song_similarities_cosine(
                transcriptions, top_k, compute_all
            )
        elif method.lower() == "lucene":
            similarities_dict = self.compute_cover_song_similarities_lucene_correct(
                transcriptions, top_k, compute_all
            )
        else:
            raise ValueError(f"Unknown method: {method}. Choose 'cosine' or 'lucene'")

        # Process each similarity matrix
        results = {}

        for key, similarities in similarities_dict.items():
            print(f"Processing {key} similarities...")

            # Convert similarities to distances (lower is better for cover detection)
            distances = 1.0 - similarities

            # For invalid transcriptions, set distances to infinity
            valid_mask = torch.tensor(
                [
                    bool(text and str(text).strip()) if isinstance(text, str) else False
                    for text in transcriptions
                ],
                dtype=torch.bool,
            )

            distances[~valid_mask, :] = float("inf")
            distances[:, ~valid_mask] = float("inf")

            # Set diagonal to infinity (don't consider self as cover)
            distances.fill_diagonal_(float("inf"))

            # Compute evaluation metrics for cover song detection
            print(
                f"Computing cover song detection metrics ({method.upper()}, {key})..."
            )
            aps, r1s, rpcs = compute_baseline(
                distances=distances,
                queries_c=clique_ids,
                queries_i=version_ids,
                candidates_c=clique_ids,
                candidates_i=version_ids,
            )

            results[key] = (aps, r1s, rpcs)

        return results


class BaselinesEvaluator:
    """
    Comprehensive baseline evaluator for version identification.

    Computes multiple baseline methods:
    - SBERT: Sentence transformer embeddings
    - TF-IDF: Cosine and Lucene-style approaches
    - Ideal: Perfect clique matching (upper bound)
    - Random: Random baseline (lower bound)
    - Modified Ideal: Perfect matching for valid transcriptions only
    - Modified Random: Upper bound for transcription-based methods

    All methods return (aps, r1s, rpcs) tuples containing:
    - aps: Average Precision scores per query
    - r1s: Recall@1 (binary: 1 if top result is correct)
    - rpcs: Rank Percentile Scores

    Args:
        model_name: Sentence transformer model name (default: all-MiniLM-L6-v2)
        device: Torch device (default: CUDA if available)
        batch_size: Batch size for SBERT encoding

    Example:
        >>> evaluator = BaselinesEvaluator()
        >>> results = evaluator.evaluate_dataset(
        ...     dataloader,
        ...     compute_baselines=['sbert', 'tfidf-cosine', 'ideal', 'random']
        ... )
        >>> # results = {'sbert': (aps, r1s, rpcs), 'ideal': (aps, r1s, rpcs), ...}
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[torch.device] = None,
        batch_size: int = 32,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = SentenceTransformer(model_name, device=self.device)
        self.batch_size = batch_size

        # Initialize paper-accurate TF-IDF evaluator for cover song detection
        self.tfidf_evaluator = TFIDFEvaluator()

        print(f"Using model: {model_name} on {self.device}")

    def encode_in_chunks(
        self, texts: List[str], chunk_size: int = 1000
    ) -> torch.Tensor:
        """Encode all texts in chunks, replacing empty/None with empty embeddings"""
        print(f"Encoding {len(texts)} transcriptions...")

        # Replace None/empty with empty string
        processed_texts = []
        for text in texts:
            if text is None or (isinstance(text, str) and not text.strip()):
                processed_texts.append("")  # Empty string for encoding
            else:
                processed_texts.append(str(text).strip())

        all_embeddings = []

        # Process in chunks
        for i in tqdm(range(0, len(processed_texts), chunk_size), desc="Encoding"):
            chunk = processed_texts[i : i + chunk_size]

            # Encode chunk
            embeddings = self.model.encode(
                chunk,
                batch_size=self.batch_size,
                convert_to_tensor=True,
                show_progress_bar=False,
                device=self.device,
            )

            # Move to CPU to save GPU memory
            all_embeddings.append(embeddings.cpu())

            # Clear GPU memory
            del embeddings
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()

        # Concatenate all chunks
        return torch.cat(all_embeddings, dim=0)

    def compute_similarities_chunked(
        self,
        query_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        chunk_size: int = 500,
    ) -> torch.Tensor:
        """Compute cosine similarities between queries and candidates in chunks"""
        print("Computing similarities...")

        num_queries = query_embeddings.shape[0]
        num_candidates = candidate_embeddings.shape[0]

        # Initialize similarity matrix
        similarities = torch.zeros((num_queries, num_candidates), dtype=torch.float32)

        # Normalize embeddings once
        query_norm = F.normalize(query_embeddings, p=2, dim=-1)
        candidate_norm = F.normalize(candidate_embeddings, p=2, dim=-1)

        # Process queries in chunks
        for q_start in tqdm(range(0, num_queries, chunk_size), desc="Query chunks"):
            q_end = min(q_start + chunk_size, num_queries)
            query_chunk = query_norm[q_start:q_end].to(self.device)

            # Process candidates in chunks
            for c_start in range(0, num_candidates, chunk_size):
                c_end = min(c_start + chunk_size, num_candidates)
                candidate_chunk = candidate_norm[c_start:c_end].to(self.device)

                # Compute cosine similarity
                sim_chunk = torch.mm(query_chunk, candidate_chunk.t())
                similarities[q_start:q_end, c_start:c_end] = sim_chunk.cpu()

                # Clear GPU memory
                del candidate_chunk, sim_chunk
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # Clear GPU memory
            del query_chunk
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return similarities

    def convert_similarities_to_distances_chunked(
        self, similarities: torch.Tensor, chunk_size: int = 1000
    ) -> torch.Tensor:
        """Convert similarities to distances in chunks to save memory"""
        print("Converting similarities to distances...")

        num_rows = similarities.shape[0]

        # Process in chunks
        for start_idx in tqdm(
            range(0, num_rows, chunk_size), desc="Converting to distances"
        ):
            end_idx = min(start_idx + chunk_size, num_rows)

            # Convert chunk in-place
            similarities[start_idx:end_idx] = 1.0 - similarities[start_idx:end_idx]

        return similarities

    def compute_ideal_baseline_distances(
        self, clique_ids: torch.Tensor, chunk_size: int = 1000
    ) -> torch.Tensor:
        """Compute ideal baseline distances: 0 for same clique, 1 for different clique"""
        print("Computing ideal baseline distances...")
        n = len(clique_ids)
        distances = torch.ones((n, n), dtype=torch.float32)

        # Process in chunks to avoid memory issues
        for i in tqdm(range(0, n, chunk_size), desc="Ideal baseline chunks"):
            i_end = min(i + chunk_size, n)

            # Process candidates in chunks
            for j in range(0, n, chunk_size):
                j_end = min(j + chunk_size, n)

                # Use broadcasting for efficient comparison
                matches = clique_ids[i:i_end, None] == clique_ids[None, j:j_end]
                distances[i:i_end, j:j_end].masked_fill_(matches, 0.0)

        return distances

    # def compute_random_baseline_distances(self, n_queries: int, n_candidates: int) -> torch.Tensor:
    #     """Compute random baseline distances"""
    #     print(f"Computing random baseline distances ({n_queries}x{n_candidates})...")
    #     return torch.rand((n_queries, n_candidates), dtype=torch.float32)
    def compute_random_baseline_distances(
        self, n_queries: int, n_candidates: int
    ) -> torch.Tensor:
        """
        Compute random baseline distances with proper seed for reproducibility
        """
        print(f"Computing random baseline distances ({n_queries}x{n_candidates})...")

        # Set seed for reproducible results
        torch.manual_seed(42)

        # Generate random distances
        distances = torch.rand((n_queries, n_candidates), dtype=torch.float32)

        # Set diagonal to infinity (don't retrieve the query itself)
        if n_queries == n_candidates:
            distances.fill_diagonal_(float("inf"))

        return distances

    # def compute_modified_ideal_baseline_distances(self, clique_ids: torch.Tensor,
    #                                             valid_mask: torch.Tensor,
    #                                             chunk_size: int = 1000) -> torch.Tensor:
    #     """Compute modified ideal baseline distances"""
    #     print("Computing modified ideal baseline distances...")
    #     n = len(clique_ids)
    #     distances = torch.ones((n, n), dtype=torch.float32)

    #     # Process in chunks to avoid memory issues
    #     for i in tqdm(range(0, n, chunk_size), desc="Modified ideal baseline chunks"):
    #         i_end = min(i + chunk_size, n)

    #         # Process candidates in chunks
    #         for j in range(0, n, chunk_size):
    #             j_end = min(j + chunk_size, n)

    #             # Check for same clique
    #             clique_matches = clique_ids[i:i_end, None] == clique_ids[None, j:j_end]

    #             # Check for valid transcriptions (both query and candidate must be valid)
    #             valid_matches = valid_mask[i:i_end, None] & valid_mask[None, j:j_end]

    #             # Combine conditions: same clique AND both valid
    #             final_matches = clique_matches & valid_matches

    #             distances[i:i_end, j:j_end].masked_fill_(final_matches, 0.0)

    #     return distances
    def compute_modified_ideal_baseline_distances(
        self, clique_ids: torch.Tensor, valid_mask: torch.Tensor, chunk_size: int = 1000
    ) -> torch.Tensor:
        """
        Compute modified ideal baseline distances:
        - Start with random distances [0,1)
        - For same-clique songs where we CAN extract transcriptions: set distance = 0 (perfect match)
        - For same-clique songs where we CANNOT extract transcriptions: keep random distance

        This simulates a scenario where lyrics-based methods work perfectly when
        transcriptions are available, showing upper bound performance.
        """
        print("Computing modified ideal baseline distances...")
        n = len(clique_ids)

        # Start with random distances [0,1)
        distances = torch.rand((n, n), dtype=torch.float32)

        # Process in chunks to avoid memory issues
        for i in tqdm(range(0, n, chunk_size), desc="Modified ideal baseline chunks"):
            i_end = min(i + chunk_size, n)

            # Process candidates in chunks
            for j in range(0, n, chunk_size):
                j_end = min(j + chunk_size, n)

                # Check for same clique (ground truth matches)
                clique_matches = clique_ids[i:i_end, None] == clique_ids[None, j:j_end]

                # Check for VALID transcriptions (songs we can extract)
                # For candidates where we can extract transcriptions
                candidate_valid = valid_mask[None, j:j_end]

                # Set distance=0 for same-clique songs where we CAN extract transcriptions
                # This gives perfect performance for songs with valid transcriptions
                perfect_matches = clique_matches & candidate_valid

                distances[i:i_end, j:j_end].masked_fill_(perfect_matches, 0.0)

        return distances

    # def compute_modified_random_baseline_distances(self, clique_ids: torch.Tensor,
    #                                          valid_transcription_mask: torch.Tensor,
    #                                          chunk_size: int = 1000) -> torch.Tensor:
    #     """
    #     Compute modified-random baseline distances: Start from random distances,
    #     then set distance=0 for candidates with valid transcriptions AND same clique matches.
    #     This creates an upper bound by giving perfect scores to lyric matches.
    #     """
    #     print("Computing modified-random baseline distances...")
    #     n = len(clique_ids)

    #     # Start with random distances
    #     distances = torch.rand((n, n), dtype=torch.float32)

    #     # Process in chunks to avoid memory issues
    #     for i in tqdm(range(0, n, chunk_size), desc="Modified-random baseline chunks"):
    #         i_end = min(i + chunk_size, n)

    #         # Process candidates in chunks
    #         for j in range(0, n, chunk_size):
    #             j_end = min(j + chunk_size, n)

    #             # Check for same clique (ground truth matches)
    #             clique_matches = clique_ids[i:i_end, None] == clique_ids[None, j:j_end]

    #             # Check for valid transcriptions (candidates with lyrics)
    #             candidate_has_lyrics = valid_transcription_mask[None, j:j_end]

    #             # Set distance=0 for candidates that have both lyrics AND are ground truth matches
    #             lyric_matches = clique_matches & candidate_has_lyrics

    #             # Apply the modification: distance=0 means perfect match (best score)
    #             distances[i:i_end, j:j_end].masked_fill_(lyric_matches, 0.0)

    #     return distances
    def compute_modified_random_baseline_distances(
        self,
        clique_ids: torch.Tensor,
        valid_transcription_mask: torch.Tensor,
        chunk_size: int = 1000,
    ) -> torch.Tensor:
        """
        Modified-random (symmetric, faster):
        - Start with random distances in [0,1)
        - For SAME-clique pairs where BOTH query and candidate have valid transcriptions, set distance = 0
        - Everything else stays random
        - Matrix is symmetric; diagonal set to +inf
        """
        print("Computing modified-random baseline distances (symmetric)...")
        n = len(clique_ids)

        # Reproducible random init
        torch.manual_seed(42)
        distances = torch.ones((n, n), dtype=torch.float32)

        for i in tqdm(range(0, n, chunk_size), desc="Modified-random (upper-tri)"):
            i_end = min(i + chunk_size, n)
            q_valid = valid_transcription_mask[i:i_end, None]  # (bi, 1)

            # Only process j >= i (upper triangle)
            for j in range(i, n, chunk_size):
                j_end = min(j + chunk_size, n)

                clique_matches = (
                    clique_ids[i:i_end, None] == clique_ids[None, j:j_end]
                )  # (bi, bj)
                c_valid = valid_transcription_mask[None, j:j_end]  # (1, bj)

                both_valid_same_clique = clique_matches & q_valid & c_valid

                # Set perfect matches to 0 in the upper block
                distances[i:i_end, j:j_end].masked_fill_(both_valid_same_clique, 0.0)

                # Mirror to lower block to keep symmetry
                if j != i:
                    distances[j:j_end, i:i_end] = distances[i:i_end, j:j_end].T

        distances += torch.rand((n, n), dtype=torch.float32)

        # No self-retrieval
        distances.fill_diagonal_(float("inf"))
        return distances

    def evaluate_dataset(
        self,
        dataloader,
        encode_chunk_size: int = 1000,
        similarity_chunk_size: int = 500,
        tfidf_top_k: int = 100,
        tfidf_method: str = "cosine",
        compute_baselines: List[str] = None,
        compute_all_tfidf: bool = True,
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Evaluate dataset with both TF-IDF approaches for cover song detection

        Args:
            compute_all_tfidf: Whether to compute TF-IDF results both with and without top-k filtering
        """
        if compute_baselines is None:
            compute_baselines = ["sbert"]

        # Extract all data from dataset
        print("Extracting data from dataset...")
        dataset = dataloader.dataset

        if hasattr(dataset, "df") and not dataset.df.empty:
            # Extract from dataframe directly
            transcription_col = f"transcription_{dataset.whisper_set}"
            clique_col = "clique_idx"
            version_col = "version_idx"

            all_transcriptions = dataset.df[transcription_col].fillna("").tolist()
            clique_ids = torch.tensor(dataset.df[clique_col].values, dtype=torch.long)
            version_ids = torch.tensor(dataset.df[version_col].values, dtype=torch.long)

            # FIXED: Use the actual enhanced validation results from the dataset
            valid_transcription_col = f"has_valid_transcription_{dataset.whisper_set}"
            if valid_transcription_col in dataset.df.columns:
                # Use the enhanced validation results
                valid_transcription_mask = torch.tensor(
                    dataset.df[valid_transcription_col].values, dtype=torch.bool
                )
                print(f"Using enhanced validation results from dataset")
                print(
                    f"Enhanced validation rate: {valid_transcription_mask.sum().item()}/{len(valid_transcription_mask)} = {valid_transcription_mask.sum().item() / len(valid_transcription_mask) * 100:.2f}%"
                )
            else:
                # Fallback to simple empty check if enhanced validation not available
                print(
                    f"Warning: Enhanced validation column '{valid_transcription_col}' not found, using simple empty check"
                )
                valid_transcription_mask = torch.tensor(
                    [bool(text and str(text).strip()) for text in all_transcriptions],
                    dtype=torch.bool,
                )
        else:
            # Extract from dataloader - this path needs to be updated too
            all_transcriptions = []
            clique_ids = []
            version_ids = []
            valid_flags = []

            for batch in tqdm(dataloader, desc="Loading data"):
                clique_batch, version_batch, _, _, _, transcriptions, valid_batch = (
                    batch
                )
                all_transcriptions.extend(transcriptions)
                clique_ids.extend(clique_batch.tolist())
                version_ids.extend(version_batch.tolist())
                # FIXED: Use the enhanced validation flags from the batch
                valid_flags.extend(valid_batch.tolist())

            clique_ids = torch.tensor(clique_ids, dtype=torch.long)
            version_ids = torch.tensor(version_ids, dtype=torch.long)
            # FIXED: Use the actual validation flags from the dataset
            valid_transcription_mask = torch.tensor(valid_flags, dtype=torch.bool)

        print(f"Total samples: {len(all_transcriptions)}")
        print(
            f"Valid transcriptions (for modified ideal): {valid_transcription_mask.sum().item()}"
        )
        print(
            f"Invalid transcriptions (for modified ideal): {(~valid_transcription_mask).sum().item()}"
        )
        print(f"Unique cliques: {len(torch.unique(clique_ids))}")

        results = {}

        # Compute TF-IDF baselines for cover song detection
        if (
            "tfidf" in compute_baselines
            or "tfidf-cosine" in compute_baselines
            or "tfidf-lucene" in compute_baselines
        ):
            # Determine which TF-IDF methods to compute
            methods_to_compute = []

            if "tfidf" in compute_baselines:
                if tfidf_method.lower() == "both":
                    methods_to_compute = ["cosine", "lucene"]
                else:
                    methods_to_compute = [tfidf_method.lower()]

            if "tfidf-cosine" in compute_baselines:
                if "cosine" not in methods_to_compute:
                    methods_to_compute.append("cosine")

            if "tfidf-lucene" in compute_baselines:
                if "lucene" not in methods_to_compute:
                    methods_to_compute.append("lucene")

            # Compute each method
            for method in methods_to_compute:
                print(f"\n--- Computing TF-IDF {method.upper()} baseline ---")
                method_results = (
                    self.tfidf_evaluator.compute_tfidf_baseline_cover_song_detection(
                        all_transcriptions,
                        clique_ids,
                        version_ids,
                        method=method,
                        top_k=tfidf_top_k,
                        compute_all=compute_all_tfidf,
                    )
                )

                # Store results with appropriate keys
                for variant, metrics in method_results.items():
                    if len(methods_to_compute) == 1 and "tfidf" in compute_baselines:
                        if variant == "top_k":
                            results["tfidf"] = metrics
                        else:  # variant == 'all'
                            results["tfidf_all"] = metrics
                    else:
                        if variant == "top_k":
                            results[f"tfidf-{method}"] = metrics
                        else:  # variant == 'all'
                            results[f"tfidf-{method}_all"] = metrics

                gc.collect()

        # Compute SBERT baseline
        if "sbert" in compute_baselines:
            print("\n--- Computing SBERT baseline ---")

            # Encode ALL transcriptions (including empty ones)
            embeddings = self.encode_in_chunks(all_transcriptions, encode_chunk_size)

            # Compute similarities
            similarities = self.compute_similarities_chunked(
                embeddings, embeddings, similarity_chunk_size
            )

            # Convert to distances (lower is better)
            distances = self.convert_similarities_to_distances_chunked(similarities)

            # For empty transcriptions, set distances to infinity
            empty_mask = torch.tensor(
                [
                    not text or not text.strip() if isinstance(text, str) else True
                    for text in all_transcriptions
                ],
                dtype=torch.bool,
            )

            distances[empty_mask, :] = float("inf")
            distances[:, empty_mask] = float("inf")

            # Compute evaluation metrics
            print("Computing SBERT metrics...")
            aps, r1s, rpcs = compute_baseline(
                distances=distances,
                queries_c=clique_ids,
                queries_i=version_ids,
                candidates_c=clique_ids,
                candidates_i=version_ids,
            )

            results["sbert"] = (aps, r1s, rpcs)

            # Clean up SBERT-specific tensors
            del embeddings, similarities, distances
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Compute Ideal baseline
        if "ideal" in compute_baselines:
            print("\n--- Computing Ideal baseline ---")
            ideal_distances = self.compute_ideal_baseline_distances(clique_ids)

            print("Computing Ideal metrics...")
            aps, r1s, rpcs = compute_baseline(
                distances=ideal_distances,
                queries_c=clique_ids,
                queries_i=version_ids,
                candidates_c=clique_ids,
                candidates_i=version_ids,
            )

            results["ideal"] = (aps, r1s, rpcs)

            del ideal_distances
            gc.collect()

        # Compute Random baseline
        if "random" in compute_baselines:
            print("\n--- Computing Random baseline ---")
            random_distances = self.compute_random_baseline_distances(
                len(clique_ids), len(clique_ids)
            )

            print("Computing Random metrics...")
            aps, r1s, rpcs = compute_baseline(
                distances=random_distances,
                queries_c=clique_ids,
                queries_i=version_ids,
                candidates_c=clique_ids,
                candidates_i=version_ids,
            )

            results["random"] = (aps, r1s, rpcs)

            del random_distances
            gc.collect()

        # Compute Modified Ideal baseline
        if "modified_ideal" in compute_baselines:
            print("\n--- Computing Modified Ideal baseline ---")
            modified_ideal_distances = self.compute_modified_ideal_baseline_distances(
                clique_ids, valid_transcription_mask
            )

            print("Computing Modified Ideal metrics...")
            aps, r1s, rpcs = compute_baseline(
                distances=modified_ideal_distances,
                queries_c=clique_ids,
                queries_i=version_ids,
                candidates_c=clique_ids,
                candidates_i=version_ids,
            )

            results["modified_ideal"] = (aps, r1s, rpcs)

            del modified_ideal_distances
            gc.collect()

        # Compute Modified Random baseline
        if "modified-random" in compute_baselines:
            print("\n--- Computing Modified Random baseline ---")
            modified_random_distances = self.compute_modified_random_baseline_distances(
                clique_ids, valid_transcription_mask
            )

            print("Computing Modified Random metrics...")
            aps, r1s, rpcs = compute_baseline(
                distances=modified_random_distances,
                queries_c=clique_ids,
                queries_i=version_ids,
                candidates_c=clique_ids,
                candidates_i=version_ids,
            )

            results["modified-random"] = (aps, r1s, rpcs)

            del modified_random_distances
            gc.collect()

        return results
