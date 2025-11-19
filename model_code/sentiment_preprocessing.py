"""
Text preprocessing and vectorization for sentiment analysis.

Implements text cleaning, tokenization, vocabulary building, and sequence conversion.
"""

import os
import re
from typing import List, Dict, Tuple, Optional
from collections import Counter

# Try to import NLTK, but handle gracefully if not available
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    
    # Try to download required NLTK data
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            try:
                nltk.data.find("tokenizers/punkt")
            except LookupError:
                try:
                    nltk.download("punkt", quiet=True)
                except Exception:
                    pass
    
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        try:
            nltk.download("stopwords", quiet=True)
        except Exception:
            pass
    
    NLTK_AVAILABLE = True
except (ImportError, Exception):
    NLTK_AVAILABLE = False
    print("Warning: NLTK not available or data download failed. Using simple tokenization.")


# Special tokens
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_IDX = 0
UNK_IDX = 1


class Vocabulary:
    """Vocabulary for mapping words to indices."""
    
    def __init__(self, min_freq: int = 1):
        """
        Initialize vocabulary.
        
        Parameters
        ----------
        min_freq : int
            Minimum frequency for a word to be included in vocabulary.
        """
        self.min_freq = min_freq
        self.word2idx: Dict[str, int] = {
            PAD_TOKEN: PAD_IDX,
            UNK_TOKEN: UNK_IDX,
        }
        self.idx2word: Dict[int, str] = {
            PAD_IDX: PAD_TOKEN,
            UNK_IDX: UNK_TOKEN,
        }
        self.word_counts: Counter = Counter()
        self._built = False
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from list of texts.
        
        Parameters
        ----------
        texts : List[str]
            List of text strings.
        """
        # Count word frequencies
        for text in texts:
            tokens = self._tokenize(text)
            self.word_counts.update(tokens)
        
        # Add words that meet minimum frequency
        for word, count in self.word_counts.items():
            if count >= self.min_freq:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        
        self._built = True
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text (used during vocab building)."""
        if NLTK_AVAILABLE:
            try:
                return word_tokenize(text.lower())
            except Exception:
                # Fallback to simple tokenization
                return self._simple_tokenize(text.lower())
        else:
            return self._simple_tokenize(text.lower())
    
    def _simple_tokenize(self, text: str) -> List[str]:
        """Simple tokenization fallback when NLTK is not available."""
        # Split on whitespace and punctuation
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text to sequence of indices.
        
        Parameters
        ----------
        text : str
            Input text.
        
        Returns
        -------
        List[int]
            Sequence of word indices.
        """
        if NLTK_AVAILABLE:
            try:
                tokens = word_tokenize(text.lower())
            except Exception:
                tokens = self._simple_tokenize(text.lower())
        else:
            tokens = self._simple_tokenize(text.lower())
        
        indices = [
            self.word2idx.get(token, UNK_IDX)
            for token in tokens
        ]
        return indices
    
    def _simple_tokenize(self, text: str) -> List[str]:
        """Simple tokenization fallback when NLTK is not available."""
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def decode(self, indices: List[int]) -> str:
        """
        Decode sequence of indices to text.
        
        Parameters
        ----------
        indices : List[int]
            Sequence of word indices.
        
        Returns
        -------
        str
            Decoded text.
        """
        words = [
            self.idx2word.get(idx, UNK_TOKEN)
            for idx in indices
            if idx != PAD_IDX
        ]
        return " ".join(words)
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.word2idx)


def clean_text(text: str, remove_stopwords: bool = False) -> str:
    """
    Clean text by removing HTML tags, converting to lowercase, etc.
    
    Parameters
    ----------
    text : str
        Input text.
    remove_stopwords : bool
        Whether to remove stopwords (default: False, as they may carry sentiment).
    
    Returns
    -------
    str
        Cleaned text.
    """
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    
    # Remove special characters but keep punctuation for sentiment
    # Keep: letters, numbers, spaces, and common punctuation
    text = re.sub(r"[^\w\s.,!?;:()'-]", "", text)
    
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    
    # Optionally remove stopwords
    if remove_stopwords and NLTK_AVAILABLE:
        try:
            stop_words = set(stopwords.words("english"))
            if NLTK_AVAILABLE:
                try:
                    tokens = word_tokenize(text.lower())
                except Exception:
                    tokens = re.findall(r'\b\w+\b', text.lower())
            else:
                tokens = re.findall(r'\b\w+\b', text.lower())
            tokens = [t for t in tokens if t not in stop_words]
            text = " ".join(tokens)
        except Exception:
            pass  # If stopwords not available, continue without removal
    
    return text


def pad_sequences(
    sequences: List[List[int]],
    max_len: Optional[int] = None,
    pad_value: int = PAD_IDX,
    truncate: bool = True,
) -> List[List[int]]:
    """
    Pad or truncate sequences to the same length.
    
    Parameters
    ----------
    sequences : List[List[int]]
        List of sequences to pad.
    max_len : Optional[int]
        Maximum length. If None, use the length of the longest sequence.
    pad_value : int
        Value to use for padding (default: PAD_IDX).
    truncate : bool
        Whether to truncate sequences longer than max_len (default: True).
    
    Returns
    -------
    List[List[int]]
        Padded sequences.
    """
    if max_len is None:
        max_len = max(len(seq) for seq in sequences) if sequences else 0
    
    padded = []
    for seq in sequences:
        if len(seq) > max_len and truncate:
            seq = seq[:max_len]
        elif len(seq) < max_len:
            seq = seq + [pad_value] * (max_len - len(seq))
        padded.append(seq)
    
    return padded


class TextPreprocessor:
    """Complete preprocessing pipeline."""
    
    def __init__(
        self,
        vocab: Optional[Vocabulary] = None,
        max_len: int = 128,
        min_freq: int = 1,
        remove_stopwords: bool = False,
    ):
        """
        Initialize preprocessor.
        
        Parameters
        ----------
        vocab : Optional[Vocabulary]
            Pre-built vocabulary. If None, will build from data.
        max_len : int
            Maximum sequence length (default: 128).
        min_freq : int
            Minimum word frequency for vocabulary (default: 1).
        remove_stopwords : bool
            Whether to remove stopwords (default: False).
        """
        self.vocab = vocab or Vocabulary(min_freq=min_freq)
        self.max_len = max_len
        self.remove_stopwords = remove_stopwords
    
    def fit(self, texts: List[str]) -> None:
        """
        Fit preprocessor on training texts (build vocabulary).
        
        Parameters
        ----------
        texts : List[str]
            Training texts.
        """
        cleaned_texts = [clean_text(text, self.remove_stopwords) for text in texts]
        self.vocab.build_vocab(cleaned_texts)
    
    def transform(self, texts: List[str]) -> List[List[int]]:
        """
        Transform texts to sequences of indices.
        
        Parameters
        ----------
        texts : List[str]
            Input texts.
        
        Returns
        -------
        List[List[int]]
            Sequences of word indices.
        """
        cleaned_texts = [clean_text(text, self.remove_stopwords) for text in texts]
        sequences = [self.vocab.encode(text) for text in cleaned_texts]
        sequences = pad_sequences(sequences, max_len=self.max_len, truncate=True)
        return sequences
    
    def fit_transform(self, texts: List[str]) -> List[List[int]]:
        """
        Fit and transform texts.
        
        Parameters
        ----------
        texts : List[str]
            Input texts.
        
        Returns
        -------
        List[List[int]]
            Sequences of word indices.
        """
        self.fit(texts)
        return self.transform(texts)


def load_glove_embeddings(
    glove_path: str,
    vocab: Vocabulary,
    embedding_dim: int = 100,
) -> Optional[Dict[str, List[float]]]:
    """
    Load GloVe embeddings from file.
    
    Parameters
    ----------
    glove_path : str
        Path to GloVe embeddings file.
    vocab : Vocabulary
        Vocabulary to match embeddings.
    embedding_dim : int
        Embedding dimension (default: 100).
    
    Returns
    -------
    Optional[Dict[str, List[float]]]
        Dictionary mapping words to embedding vectors, or None if file not found.
    """
    if not os.path.exists(glove_path):
        return None
    
    if not hasattr(vocab, 'word2idx') or not vocab.word2idx:
        print("Warning: Vocabulary not built. Cannot load GloVe embeddings.")
        return None
    
    embeddings = {}
    try:
        with open(glove_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != embedding_dim + 1:
                    continue
                word = parts[0]
                if word in vocab.word2idx:
                    vector = [float(x) for x in parts[1:]]
                    embeddings[word] = vector
    except Exception as e:
        print(f"Error loading GloVe embeddings: {e}")
        return None
    
    return embeddings


def create_embedding_matrix(
    vocab: Vocabulary,
    embeddings: Dict[str, List[float]],
    embedding_dim: int,
) -> List[List[float]]:
    """
    Create embedding matrix from vocabulary and pre-trained embeddings.
    
    Parameters
    ----------
    vocab : Vocabulary
        Vocabulary.
    embeddings : Dict[str, List[float]]
        Pre-trained word embeddings.
    embedding_dim : int
        Embedding dimension.
    
    Returns
    -------
    List[List[float]]
        Embedding matrix of shape (vocab_size, embedding_dim).
    """
    import numpy as np
    
    vocab_size = len(vocab)
    embedding_matrix = np.random.normal(0, 0.1, (vocab_size, embedding_dim))
    
    # Set embeddings for words in vocabulary
    for word, idx in vocab.word2idx.items():
        if word in embeddings:
            embedding_matrix[idx] = embeddings[word]
        elif word == PAD_TOKEN:
            embedding_matrix[idx] = np.zeros(embedding_dim)
    
    return embedding_matrix.tolist()


if __name__ == "__main__":
    # Test preprocessing
    texts = [
        "Apple stock surges on strong earnings report!",
        "Company faces challenges in competitive market.",
        "No significant changes reported this quarter.",
    ]
    
    preprocessor = TextPreprocessor(max_len=20)
    sequences = preprocessor.fit_transform(texts)
    
    print("Vocabulary size:", len(preprocessor.vocab))
    print("Sequences:")
    for text, seq in zip(texts, sequences):
        print(f"  {text[:50]} -> {seq[:10]}...")

