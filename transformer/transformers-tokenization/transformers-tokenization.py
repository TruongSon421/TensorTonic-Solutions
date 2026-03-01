import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # YOUR CODE HERE
        # 1. Add special tokens to the vocabulary first
        special_tokens = [self.pad_token, self.unk_token, self.bos_token,self.eos_token]

        for token in special_tokens:
            if token not in self.word_to_id:
                self.word_to_id[token] = self.vocab_size
                self.id_to_word[self.vocab_size] = token
                self.vocab_size += 1

        # 2. Iterate through texts to add unique words

        for text in texts:
            words = text.split()
            for word in words:
                if word not in self.word_to_id:
                    self.word_to_id[word] = self.vocab_size
                    self.id_to_word[self.vocab_size] = word
                    self.vocab_size += 1
            
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        # YOUR CODE HERE
        unk_id = self.word_to_id[self.unk_token]

        words = text.split()

        token_ids = [self.word_to_id.get(word,unk_id) for word in words]

        return token_ids
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        # YOUR CODE HERE
        words = [self.id_to_word.get(token_id, self.unk_token) for token_id in ids]
        return " ".join(words)