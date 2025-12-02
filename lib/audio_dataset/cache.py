"""
RAM-based cache for transcriptions with disk persistence.
"""
import os
import pickle
import glob
from tqdm import tqdm
from .validator import TranscriptionValidator


class TranscriptionCache:
    """RAM-based cache for transcriptions with disk persistence"""
    def __init__(self, data_folder, dataset_name):
        self.data_folder = data_folder
        self.dataset_name = dataset_name
        self.cache_dir = os.path.join(data_folder, f"{dataset_name}-transcription-cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.transcription_cache = {}  # In-memory cache

    def get_cache_file(self, whisper_set, split="all"):
        """Get cache file path for a specific whisper set and split"""
        cache_id = f"{self.dataset_name}_{whisper_set}_{split}"
        return os.path.join(self.cache_dir, f"{cache_id}_cache.pkl")

    def load_disk_cache(self, whisper_set, split="all"):
        """Load transcriptions from disk cache into memory"""
        cache_file = self.get_cache_file(whisper_set, split)
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    self.transcription_cache[whisper_set] = pickle.load(f)
                print(f"Loaded cache for {whisper_set} with {len(self.transcription_cache[whisper_set])} entries")
                return True
            except Exception as e:
                print(f"Error loading cache: {e}")
        return False

    def save_disk_cache(self, whisper_set, split="all"):
        """Save in-memory cache to disk"""
        if whisper_set in self.transcription_cache:
            cache_file = self.get_cache_file(whisper_set, split)
            with open(cache_file, 'wb') as f:
                pickle.dump(self.transcription_cache[whisper_set], f)
            print(f"Cache saved to {cache_file}") 

    def build_index(self, whisper_set):
        """Load all transcriptions for a specific whisper set into memory"""
        print(f"Building index for {whisper_set}...")

        # Initialize cache for this whisper set
        if whisper_set not in self.transcription_cache:
            self.transcription_cache[whisper_set] = {}

        # Get base path and pattern based on dataset
        if self.dataset_name == "lyric-covers":
            pattern = os.path.join(self.data_folder, "LyricCovers-transcriptions",
                                   "transcriptions", "*", f"{self.dataset_name}_{whisper_set}.txt")
            print(pattern)

        elif self.dataset_name == "shs":
            pattern = os.path.join(self.data_folder, "SHS100K-transcriptions",
                                   "transcriptions", "*", "*", f"{self.dataset_name}_{whisper_set}.txt")
        elif self.dataset_name == "discogs-vi":
            pattern = os.path.join(self.data_folder, "DiscogsVI-transcriptions",
                                   "transcriptions", "*", "*", f"{self.dataset_name}_{whisper_set}.txt")
        else:
            print(f"Unsupported dataset: {self.dataset_name}")
            return self.transcription_cache[whisper_set]

        # Load all matching files
        for path in tqdm(glob.glob(pattern)):
            components = path.split(os.sep)
            # Extract key based on dataset type
            if self.dataset_name == "lyric-covers":
                key = components[-2]  # song_id
            elif self.dataset_name == "shs":
                key = components[-2]  # set_id-ver_id
            elif self.dataset_name == "discogs-vi":
                key = f"{components[-3]}/{components[-2]}"  # base_filename


            # Read and store in memory
            try:
                with open(path, "r") as f:
                    self.transcription_cache[whisper_set][key] = f.read()
            except Exception:
                continue

        print(f"Loaded {len(self.transcription_cache[whisper_set])} transcriptions")
        return self.transcription_cache[whisper_set]

    def apply_to_dataframe(self, df, whisper_sets, rebuild_cache=False, split="all"):
        """Apply transcriptions to dataframe from memory cache with enhanced validation"""
        result_df = df.copy()

        for whisper_set in whisper_sets:
            # Load or build cache
            if not rebuild_cache and self.load_disk_cache(whisper_set, split):
                pass  # Cache loaded successfully
            else:
                self.build_index(whisper_set)
                self.save_disk_cache(whisper_set, split)

            print(f"Applying transcriptions for {whisper_set}...")
            transcription_column = f"transcription_{whisper_set}"

            # Create mapping function that handles all dataset types
            if self.dataset_name == "lyric-covers":
                # Fixed: Use direct assignment instead of inplace operation
                result_df[transcription_column] = result_df["id"].astype(str).map(
                    self.transcription_cache[whisper_set]).fillna("")
            elif self.dataset_name == "shs":
                # Create composite key for SHS
                result_df['temp_key'] = result_df.apply(
                    lambda row: f"{str(row['set_id'])}-{str(row['ver_id'])}", axis=1)
                result_df[transcription_column] = result_df['temp_key'].map(
                    self.transcription_cache[whisper_set]).fillna("")
                result_df.drop('temp_key', axis=1, inplace=True)
            elif self.dataset_name == "discogs-vi":
                result_df[transcription_column] = result_df["base_filename"].map(
                    self.transcription_cache[whisper_set]).fillna("")

            # Enhanced validation column name
            valid_transcription_column = f"has_valid_transcription_{whisper_set}"

            # Initialize enhanced validator
            validator = TranscriptionValidator(
                min_words=10,
                max_repetition_ratio=0.6,  # Allow up to 60% repetition
                min_unique_bigrams=3,
                min_unique_trigrams=2
            )

            # Apply enhanced validation
            print(f"Applying enhanced transcription validation for {whisper_set}...")
            result_df[valid_transcription_column] = result_df[transcription_column].apply(
                validator.is_valid_transcription
            )

            # Create detailed validation column for debugging (optional)
            validation_details_column = f"transcription_validation_details_{whisper_set}"
            result_df[validation_details_column] = result_df[transcription_column].apply(
                validator.get_validation_details
            )

            # Report detailed statistics
            empty_count = (result_df[transcription_column] == "").sum()
            valid_count = result_df[valid_transcription_column].sum()
            invalid_count = len(result_df) - empty_count - valid_count

            print(f"Enhanced transcription validation results for {whisper_set}:")
            print(f"  Total transcriptions: {len(result_df)}")
            print(f"  Empty transcriptions: {empty_count}")
            print(f"  Valid transcriptions: {valid_count}")
            print(f"  Invalid transcriptions: {invalid_count}")
            print(f"  Validation rate: {valid_count/len(result_df)*100:.2f}%")

            # Print breakdown of validation issues
            if invalid_count > 0:
                print(f"\nValidation issue breakdown for {whisper_set}:")
                all_issues = []
                for details in result_df[validation_details_column]:
                    if isinstance(details, dict) and not details.get('is_valid', True):
                        all_issues.extend(details.get('issues', []))
                
                if all_issues:
                    from collections import Counter
                    issue_counts = Counter(all_issues)
                    for issue, count in issue_counts.items():
                        print(f"  {issue.replace('_', ' ').title()}: {count} transcriptions")
                else:
                    print("  No specific issues identified (transcriptions may be empty)")

            # Report basic statistics (keep the original reporting for compatibility)
            total_with_transcriptions = len(result_df) - empty_count
            print(f"Found {total_with_transcriptions} transcriptions out of {len(result_df)} rows")
            print(f"Found {valid_count} valid transcriptions (enhanced criteria) out of {len(result_df)} rows")

        return result_df

