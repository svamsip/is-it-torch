"""Tokenizer using Byte Pair Encoding (BPE) algorithm."""

import json
import re
from typing import List, Dict, Tuple


class BaseTokenizer:
    """Base class for tokenizers."""

    def __init__(self):
        self.merge_rules: Dict[Tuple[str, str], int] = {}
        self.pattern: str = ""
        self.special_tokens: Dict[str, int] = {}
        self.vocab: Dict[str, int] = {}

    def tokenize(self, text: str) -> List[str]:
        raise NotImplementedError("Subclasses should implement this method.")

    def detokenize(self, tokens: List[str]) -> str:
        raise NotImplementedError("Subclasses should implement this method.")


class BPETokenizer(BaseTokenizer):
    """Tokenizer using Byte Pair Encoding (BPE) algorithm."""

    def __init__(self):
        super().__init__()

    def train(self, text_data: str, vocab_size: int) -> None:
        """Train the tokenizer on the given text data."""
        vocab_size = max(256, vocab_size)  # Ensure minimum vocab size for UTF-8
        num_merges = vocab_size - 256

        # Preprocess text data
        text_data = text_data.lower().replace("\n", " <eos> ").split()

        # Initialize vocabulary with characters and words
        vocab = {}
        for word in text_data:
            for char in word:
                vocab[char] = vocab.get(char, 0) + 1
            vocab[word] = vocab.get(word, 0) + 1

        # Perform merges to create merge rules
        self.merge_rules = {}
        for _ in range(num_merges):
            pairs = self._get_adjacent_pairs(vocab)
            if not pairs:
                break

            # Find the most frequent pair
            best_pair = max(pairs, key=pairs.get)
            self.merge_rules[best_pair] = len(self.merge_rules) + 256

            # Merge the best pair in the vocabulary
            vocab = self._merge_pair_in_vocab(vocab, best_pair)

        # Finalize vocabulary and special tokens
        self._finalize_vocab(vocab)

        print(f"Training complete. Vocabulary size: {len(self.vocab)}")

    def _get_adjacent_pairs(self, vocab: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        """Get adjacent character pairs and their frequencies."""
        pairs = {}
        for word, freq in vocab.items():
            if len(word) < 2:
                continue
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pairs[pair] = pairs.get(pair, 0) + freq
        return pairs

    def _merge_pair_in_vocab(self, vocab: Dict[str, int], pair: Tuple[str, str]) -> Dict[str, int]:
        """Merge the given pair in the vocabulary."""
        # new_vocab = {}
        # pair_str = "".join(pair)
        # for word, freq in vocab.items():
        #     new_word = re.sub(r"(?<!\S)" + re.escape(pair_str) + r"(?!\S)", pair_str, word)
        #     new_vocab[new_word] = new_vocab.get(new_word, 0) + freq
        # return new_vocab

        new_vocab = {}
        pair_str = "".join(pair)
        pattern = re.escape(pair[0]) + " " + re.escape(pair[1])
        for word, freq in vocab.items():
            new_word = re.sub(pattern, pair_str, word)
            new_vocab[new_word] = new_vocab.get(new_word, 0) + freq
        return new_vocab

    def _finalize_vocab(self, vocab: Dict[str, int]) -> None:
        """Finalize the vocabulary and special tokens."""
        self.vocab = {word: idx for idx, (word, _) in enumerate(vocab.items(), start=256)}
        self.special_tokens = {"<pad>": 0, "<unk>": 1, "<eos>": 2, "<bos>": 3}

        # Add special tokens to the vocabulary
        for token, idx in self.special_tokens.items():
            self.vocab[token] = idx

        # Ensure the vocabulary is sorted by index
        self.vocab = dict(sorted(self.vocab.items(), key=lambda item: item[1]))

        # Update the pattern for tokenization
        self.pattern = "|".join(sorted(self.vocab.keys(), key=lambda x: -len(x)))

    def tokenize(self, text: str) -> List[str]:
        """Tokenize the given text while preserving word boundaries."""
        if not self.pattern:
            raise ValueError("Tokenizer is not trained. Call train() first.")

        # Remove the lowercasing so that case is preserved; keep <eos> replacement.
        text = text.replace("\n", " <eos> ")

        # Split the text into words based on whitespace.
        words = text.split()
        tokens = []

        # Tokenize each word separately.
        for word in words:
            tokenized_word = []
            while word:
                match = re.match(self.pattern, word)
                if match:
                    token = match.group(0)
                    tokenized_word.append(token)
                    word = word[len(token) :]
                else:
                    tokenized_word.append(word[0])
                    word = word[1:]
            # Join the subword tokens back into a word.
            tokens.append("".join(tokenized_word))

        return tokens

    def detokenize(self, tokens: List[str]) -> str:
        """Detokenize the given list of tokens while restoring spaces."""
        if not tokens:
            return ""

        # Join tokens with a space, so that word boundaries are restored.
        text = " ".join(tokens)

        # Replace special tokens with their desired representations.
        text = text.replace("<eos>", "\n").strip()
        return text

    def get_vocab(self) -> Dict[str, int]:
        """Returns the vocabulary of the tokenizer."""
        return self.vocab

    def get_special_tokens(self) -> Dict[str, int]:
        """Returns the special tokens of the tokenizer."""
        return self.special_tokens

    def get_merge_rules(self) -> Dict[Tuple[str, str], int]:
        """Returns the merge rules of the tokenizer."""
        return self.merge_rules

    def save(self, filepath: str) -> None:
        """Saves the tokenizer's vocabulary and merge rules to a file."""
        data = {
            "vocab": self.vocab,
            "special_tokens": self.special_tokens,
            "merge_rules": {f"{k[0]}_{k[1]}": v for k, v in self.merge_rules.items()},
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Tokenizer saved to {filepath}")

    def load(self, filepath: str) -> "BPETokenizer":
        """Loads the tokenizer's vocabulary and merge rules from a file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.vocab = data.get("vocab", {})
            self.special_tokens = data.get("special_tokens", {})
            self.merge_rules = data.get("merge_rules", {})

        # Rebuild the pattern for tokenization
        self.pattern = "|".join(sorted(map(re.escape, self.vocab.keys()), key=lambda x: -len(x)))

        print(f"Tokenizer loaded from {filepath}. Vocabulary size: {len(self.vocab)}")
        return self


if __name__ == "__main__":

    sample_text = """
    This is a larger dataset for training the Byte Pair Encoding tokenizer.
    Byte Pair Encoding works by merging frequent pairs of characters or subwords.
    It is widely used in natural language processing tasks.
    """
    tokenizer = BPETokenizer()
    tokenizer.train(sample_text, vocab_size=1000)

    tokens = tokenizer.tokenize("Hello world! This is a test.")
    print("Tokens:", tokens, sep="\n")

    detokenized_text = tokenizer.detokenize(tokens)
    print("Detokenized Text:", detokenized_text, sep="\n")

    tokenizer.save("bpe_tokenizer.json")

    loaded_tokenizer = BPETokenizer().load("bpe_tokenizer.json")
    loaded_tokens = loaded_tokenizer.tokenize("Byte Pair Encoding is great!")
    print("Loaded Tokens:", loaded_tokens, sep="\n")
    loaded_detokenized_text = loaded_tokenizer.detokenize(loaded_tokens)
    print("Loaded Detokenized Text:", loaded_detokenized_text, sep="\n")
    assert loaded_detokenized_text == "Byte Pair Encoding is great!", "Loaded tokenizer did not work correctly!"
    print("Loaded tokenizer works correctly!", sep="\n")
