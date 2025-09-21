# Custom Subword Tokenizers

This repository contains implementations of four subword tokenization algorithms: Byte-Pair Encoding (BPE), WordPiece, Unigram Language Model, and a SentencePiece-style BPE. These tokenizers are implemented from scratch in Python, using only the standard library and NumPy.

## Tokenizers

### 1. Byte-Pair Encoding (BPE)

**Description:** A greedy, frequency-based algorithm that iteratively merges the most frequent adjacent character pairs to form new subword units.

**How to use:**

```bash
python byte-pair_tokenizer.py --train <train_file.txt> --input <input_file.txt> --vocab_size <size>
```

### 2. WordPiece

**Description:**  
A likelihood-driven algorithm that iteratively merges the most frequent adjacent token pairs that maximise the corpus log-likelihood under a unigram language model. Uses the `##` prefix for non-initial subwords.

**How to use:**
```bash
python word-piece_tokenizer.py --train <train_file.txt> --input <input_file.txt> --vocab_size <size>
```

### 3. Unigram Language Model

**Description:**  
A probabilistic model that starts with a large candidate vocabulary and iteratively prunes tokens that contribute least to the corpus log-likelihood. Allows for multiple possible segmentations.

**How to use:**
```bash
python unigram_tokenizer.py --train <train_file.txt>
```

### 4. SentencePiece-style BPE

**Description:**  
An adaptation of the BPE algorithm that operates on a whitespace-free, normalised text string using a special marker (`▁`) to represent spaces. Includes Unicode NFKC normalisation and lowercasing.

**How to use:**
```bash
python sentence-piece_tokenizer.py --train <train_file.txt> --input <input_file.txt> --vocab_size <size>
```
