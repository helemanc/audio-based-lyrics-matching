"""
Transcription validation logic for detecting invalid transcriptions.
"""
import re
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from collections import Counter
import nltk

nltk.download('punkt_tab')


class TranscriptionValidator:
    """
    Enhanced transcription validator for detecting invalid transcriptions
    """
    
    def __init__(self, min_words=10, max_repetition_ratio=0.7, min_unique_bigrams=3, min_unique_trigrams=2):
        """
        Initialize validator with configurable thresholds
        
        Args:
            min_words: Minimum number of words required
            max_repetition_ratio: Maximum ratio of repeated n-grams (0.7 = 70%)
            min_unique_bigrams: Minimum number of unique bigrams required
            min_unique_trigrams: Minimum number of unique trigrams required
        """
        self.min_words = min_words
        self.max_repetition_ratio = max_repetition_ratio
        self.min_unique_bigrams = min_unique_bigrams
        self.min_unique_trigrams = min_unique_trigrams
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text for analysis"""
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove timestamps and annotations
        text = re.sub(r'\[\d+:\d+\]', '', text)  # Remove [mm:ss] timestamps
        text = re.sub(r'\(.*?\)', '', text)      # Remove parenthetical annotations
        text = re.sub(r'\[.*?\]', '', text)      # Remove bracketed annotations
        
        # Remove excessive filler words
        excessive_fillers = r'\b(um|uh|ah|hmm|er|eh|mm)\b'
        text = re.sub(excessive_fillers, ' ', text)
        
        # Clean up punctuation but preserve apostrophes
        text = re.sub(r"[^\w\s']", ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def is_empty_or_too_short(self, text: str) -> bool:
        """Check if text is empty or too short"""
        cleaned_text = self.clean_text(text)
        if not cleaned_text:
            return True
        
        try:
            words = word_tokenize(cleaned_text)
            return len(words) < self.min_words
        except:
            # If tokenization fails, use simple split
            words = cleaned_text.split()
            return len(words) < self.min_words
    
    def is_only_symbols(self, text: str) -> bool:
        """Check if text contains only symbols/punctuation"""
        if not text or not isinstance(text, str):
            return True
        
        # Remove whitespace and check if anything remains
        text_no_space = re.sub(r'\s+', '', text)
        if not text_no_space:
            return True
        
        # Check if text contains only punctuation and symbols
        # Keep only alphanumeric characters and check if any remain
        alphanumeric_only = re.sub(r'[^a-zA-Z0-9]', '', text)
        
        # If less than 5 alphanumeric characters, consider it only symbols
        return len(alphanumeric_only) < 5
    
    def is_musical_content(self, text: str) -> bool:
        """
        Check if text contains primarily musical content that should be filtered out
        """
        if not text or not isinstance(text, str):
            return False
        
        # Convert to lowercase for pattern matching
        text_lower = text.lower()
        
        # Musical notation symbols (Unicode musical symbols)
        musical_symbols = r'[♪♫♬♩♭♮♯𝄞𝄢𝄪𝄫]'
        
        # Common musical annotations that Whisper might produce
        musical_annotations = [
            r'\(music\s*playing\)',
            r'\[music\]',
            r'\(music\)',
            r'\[music\s*playing\]',
            r'\(instrumental\)',
            r'\[instrumental\]',
            r'\(singing\)',
            r'\[singing\]',
            r'\(humming\)',
            r'\[humming\]',
            r'\(whistling\)',
            r'\[whistling\]',
            r'\(melody\)',
            r'\[melody\]',
            r'\(musical\s*interlude\)',
            r'\[musical\s*interlude\]'
        ]
        
        # Check for musical symbols
        if re.search(musical_symbols, text):
            # If text is mostly musical symbols, consider it musical content
            text_no_symbols = re.sub(musical_symbols, '', text)
            text_no_space = re.sub(r'\s+', '', text_no_symbols)
            if len(text_no_space) < 10:  # Very little non-musical content
                return True
        
        # Check for musical annotations
        for pattern in musical_annotations:
            if re.search(pattern, text_lower):
                return True
        
        # Check if text is primarily "la la la", "na na na", etc.
        repetitive_musical_patterns = [
            r'\b(la\s+){3,}',           # "la la la la..."
            r'\b(na\s+){3,}',           # "na na na na..."
            r'\b(da\s+){3,}',           # "da da da da..."
            r'\b(tra\s+){3,}',          # "tra tra tra tra..."
            r'\b(do\s+){3,}',           # "do do do do..."
            r'\b(re\s+){3,}',           # "re re re re..."
            r'\b(mi\s+){3,}',           # "mi mi mi mi..."
            r'\b(fa\s+){3,}',           # "fa fa fa fa..."
            r'\b(so\s+){3,}',           # "so so so so..."
            r'\b(ti\s+){3,}',           # "ti ti ti ti..."
            r'\b(doo\s+){3,}',          # "doo doo doo..."
            r'\b(bah\s+){3,}',          # "bah bah bah..."
        ]
        
        for pattern in repetitive_musical_patterns:
            if re.search(pattern, text_lower):
                return True
        
        # Check if text is mostly composed of musical syllables
        words = re.findall(r'\b\w+\b', text_lower)
        if words:
            musical_syllables = {'la', 'na', 'da', 'tra', 'do', 're', 'mi', 'fa', 'so', 'ti', 'doo', 'bah', 'hmm', 'mm'}
            musical_word_count = sum(1 for word in words if word in musical_syllables)
            
            # If more than 70% of words are musical syllables, consider it musical
            if len(words) >= 3 and (musical_word_count / len(words)) > 0.7:
                return True
        
        return False
    
    def has_excessive_repetition(self, text: str) -> bool:
        """
        Check if text has excessive repetition of bigrams or trigrams
        This catches cases like "thank you thank you thank you..." repeated
        """
        cleaned_text = self.clean_text(text)
        if not cleaned_text:
            return True
        
        try:
            words = word_tokenize(cleaned_text)
        except:
            words = cleaned_text.split()
        
        if len(words) < 4:  # Too short to analyze n-grams meaningfully
            return False
        
        # Check bigrams
        bigrams = list(ngrams(words, 2))
        if len(bigrams) >= 2:
            bigram_counts = Counter(bigrams)
            most_common_bigram_count = bigram_counts.most_common(1)[0][1]
            bigram_repetition_ratio = most_common_bigram_count / len(bigrams)
            
            # Check if we have too few unique bigrams or too much repetition
            unique_bigrams = len(set(bigrams))
            if (unique_bigrams < self.min_unique_bigrams or 
                bigram_repetition_ratio > self.max_repetition_ratio):
                return True
        
        # Check trigrams if we have enough words
        if len(words) >= 6:
            trigrams = list(ngrams(words, 3))
            if len(trigrams) >= 2:
                trigram_counts = Counter(trigrams)
                most_common_trigram_count = trigram_counts.most_common(1)[0][1]
                trigram_repetition_ratio = most_common_trigram_count / len(trigrams)
                
                # Check if we have too few unique trigrams or too much repetition
                unique_trigrams = len(set(trigrams))
                if (unique_trigrams < self.min_unique_trigrams or 
                    trigram_repetition_ratio > self.max_repetition_ratio):
                    return True
        
        return False
    
    def has_repeated_phrases(self, text: str) -> bool:
        """
        Check for repeated phrases that might indicate transcription errors
        """
        cleaned_text = self.clean_text(text)
        if not cleaned_text:
            return True
        
        # Split into sentences or chunks
        sentences = re.split(r'[.!?]+', cleaned_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return False
        
        # Check for identical or very similar sentences
        sentence_counts = Counter(sentences)
        total_sentences = len(sentences)
        
        for sentence, count in sentence_counts.items():
            if count / total_sentences > 0.5:  # More than 50% are the same sentence
                return True
        
        return False
    
    def is_valid_transcription(self, text: str) -> bool:
        """
        Main validation function that combines all checks
        
        Returns:
            bool: True if transcription is valid, False otherwise
        """
        if self.is_empty_or_too_short(text):
            return False
        
        if self.is_only_symbols(text):
            return False
        
        if self.is_musical_content(text):
            return False
        
        if self.has_excessive_repetition(text):
            return False
        
        if self.has_repeated_phrases(text):
            return False
        
        return True
    
    def get_validation_details(self, text: str) -> dict:
        """
        Get detailed validation results for debugging
        
        Returns:
            dict: Dictionary with validation details
        """
        details = {
            'is_valid': True,
            'issues': [],
            'text_length': len(text) if text else 0,
            'cleaned_text': self.clean_text(text)
        }
        
        if self.is_empty_or_too_short(text):
            details['is_valid'] = False
            details['issues'].append('empty_or_too_short')
        
        if self.is_only_symbols(text):
            details['is_valid'] = False
            details['issues'].append('only_symbols')
        
        if self.is_musical_content(text):
            details['is_valid'] = False
            details['issues'].append('musical_content')
        
        if self.has_excessive_repetition(text):
            details['is_valid'] = False
            details['issues'].append('excessive_repetition')
        
        if self.has_repeated_phrases(text):
            details['is_valid'] = False
            details['issues'].append('repeated_phrases')
        
        return details
