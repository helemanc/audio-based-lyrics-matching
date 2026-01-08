"""
Transcription validation for detecting invalid/low-quality transcriptions.

Filters out instrumental tracks, musical content, and repetitive transcriptions
using n-gram analysis and pattern matching.
"""

import re
from typing import Dict, List
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from collections import Counter
import nltk

nltk.download('punkt_tab', quiet=True)


class TranscriptionValidator:
    """
    Validates transcription quality for filtering invalid content.
    
    Detects:
    - Empty/too short transcriptions
    - Musical content (instrumental tracks)
    - Excessive repetition (n-gram analysis)
    - Repeated phrases
    
    Args:
        min_words: Minimum word count
        max_repetition_ratio: Max ratio of repeated n-grams (0.0-1.0)
        min_unique_bigrams: Min unique bigrams required
        min_unique_trigrams: Min unique trigrams required
    
    Example:
        >>> validator = TranscriptionValidator(min_words=10, max_repetition_ratio=0.6)
        >>> is_valid = validator.is_valid_transcription(text)
    """
    
    def __init__(
        self,
        min_words: int = 10,
        max_repetition_ratio: float = 0.7,
        min_unique_bigrams: int = 3,
        min_unique_trigrams: int = 2
    ) -> None:
        self.min_words = min_words
        self.max_repetition_ratio = max_repetition_ratio
        self.min_unique_bigrams = min_unique_bigrams
        self.min_unique_trigrams = min_unique_trigrams
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Removes: timestamps, annotations, excessive fillers, extra whitespace
        
        Args:
            text: Raw transcription text
        
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        text = text.lower()
        
        # Remove timestamps and annotations
        text = re.sub(r'\[\d+:\d+\]', '', text)
        text = re.sub(r'\(.*?\)', '', text)
        text = re.sub(r'\[.*?\]', '', text)
        
        # Remove excessive fillers
        text = re.sub(r'\b(um|uh|ah|hmm|er|eh|mm)\b', ' ', text)
        
        # Clean punctuation (preserve apostrophes)
        text = re.sub(r"[^\w\s']", ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def is_empty_or_too_short(self, text: str) -> bool:
        """Check if text is empty or below minimum word count."""
        cleaned = self.clean_text(text)
        if not cleaned:
            return True
        
        try:
            words = word_tokenize(cleaned)
        except:
            words = cleaned.split()
        
        return len(words) < self.min_words
    
    def is_only_symbols(self, text: str) -> bool:
        """Check if text contains only symbols/punctuation (<5 alphanumeric chars)."""
        if not text or not isinstance(text, str):
            return True
        
        alphanumeric = re.sub(r'[^a-zA-Z0-9]', '', text)
        return len(alphanumeric) < 5
    
    def is_musical_content(self, text: str) -> bool:
        """
        Detect musical content that should be filtered.
        
        Checks for:
        - Musical notation symbols
        - Musical annotations ([music], [instrumental], etc.)
        - Repetitive musical patterns (la la la, na na na, etc.)
        - High ratio of musical syllables
        
        Args:
            text: Transcription text
        
        Returns:
            True if primarily musical content
        """
        if not text or not isinstance(text, str):
            return False
        
        text_lower = text.lower()
        
        # Musical symbols
        musical_symbols = r'[♪♫♬♩♭♮♯𝄞𝄢𝄪𝄫]'
        if re.search(musical_symbols, text):
            text_no_symbols = re.sub(musical_symbols, '', text)
            if len(re.sub(r'\s+', '', text_no_symbols)) < 10:
                return True
        
        # Musical annotations
        musical_annotations = [
            r'\(music\s*playing\)', r'\[music\]', r'\(music\)',
            r'\[instrumental\]', r'\(instrumental\)',
            r'\[singing\]', r'\(singing\)',
            r'\[humming\]', r'\(humming\)',
            r'\[melody\]', r'\(melody\)'
        ]
        
        for pattern in musical_annotations:
            if re.search(pattern, text_lower):
                return True
        
        # Repetitive musical patterns
        repetitive_patterns = [
            r'\b(la\s+){3,}', r'\b(na\s+){3,}', r'\b(da\s+){3,}',
            r'\b(tra\s+){3,}', r'\b(do\s+){3,}', r'\b(doo\s+){3,}'
        ]
        
        for pattern in repetitive_patterns:
            if re.search(pattern, text_lower):
                return True
        
        # Check ratio of musical syllables
        words = re.findall(r'\b\w+\b', text_lower)
        if words:
            musical_syllables = {
                'la', 'na', 'da', 'tra', 'do', 're', 'mi', 'fa', 'so', 'ti', 'doo', 'bah'
            }
            musical_count = sum(1 for w in words if w in musical_syllables)
            
            if len(words) >= 3 and (musical_count / len(words)) > 0.7:
                return True
        
        return False
    
    def has_excessive_repetition(self, text: str) -> bool:
        """
        Check for excessive n-gram repetition.
        
        Uses bigrams and trigrams to detect patterns like:
        "thank you thank you thank you..."
        
        Args:
            text: Transcription text
        
        Returns:
            True if repetition exceeds thresholds
        """
        cleaned = self.clean_text(text)
        if not cleaned:
            return True
        
        try:
            words = word_tokenize(cleaned)
        except:
            words = cleaned.split()
        
        if len(words) < 4:
            return False
        
        # Check bigrams
        bigrams_list = list(ngrams(words, 2))
        if len(bigrams_list) >= 2:
            bigram_counts = Counter(bigrams_list)
            most_common_count = bigram_counts.most_common(1)[0][1]
            repetition_ratio = most_common_count / len(bigrams_list)
            unique_bigrams = len(set(bigrams_list))
            
            if (unique_bigrams < self.min_unique_bigrams or
                repetition_ratio > self.max_repetition_ratio):
                return True
        
        # Check trigrams
        if len(words) >= 6:
            trigrams_list = list(ngrams(words, 3))
            if len(trigrams_list) >= 2:
                trigram_counts = Counter(trigrams_list)
                most_common_count = trigram_counts.most_common(1)[0][1]
                repetition_ratio = most_common_count / len(trigrams_list)
                unique_trigrams = len(set(trigrams_list))
                
                if (unique_trigrams < self.min_unique_trigrams or
                    repetition_ratio > self.max_repetition_ratio):
                    return True
        
        return False
    
    def has_repeated_phrases(self, text: str) -> bool:
        """
        Check for repeated phrases (>50% identical sentences).
        
        Args:
            text: Transcription text
        
        Returns:
            True if excessive phrase repetition detected
        """
        cleaned = self.clean_text(text)
        if not cleaned:
            return True
        
        sentences = re.split(r'[.!?]+', cleaned)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return False
        
        sentence_counts = Counter(sentences)
        total = len(sentences)
        
        for sentence, count in sentence_counts.items():
            if count / total > 0.5:
                return True
        
        return False
    
    def is_valid_transcription(self, text: str) -> bool:
        """
        Main validation function combining all checks.
        
        Args:
            text: Transcription text to validate
        
        Returns:
            True if transcription passes all checks
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
    
    def get_validation_details(self, text: str) -> Dict[str, any]:
        """
        Get detailed validation results for debugging.
        
        Args:
            text: Transcription text
        
        Returns:
            Dict with validation details:
                - is_valid: bool
                - issues: List[str]
                - text_length: int
                - cleaned_text: str
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