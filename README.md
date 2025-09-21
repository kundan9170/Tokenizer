# Custom Subword Tokenizers

This repository contains implementations of four subword tokenization algorithms: Byte-Pair Encoding (BPE), WordPiece, Unigram Language Model, and a SentencePiece-style BPE. These tokenizers are implemented from scratch in Python, using only the standard library and NumPy.

## Tokenizers

### 1. Byte-Pair Encoding (BPE)

**Description:** A greedy, frequency-based algorithm that iteratively merges the most frequent adjacent character pairs to form new subword units.

**How to use:**

```bash
python <rollno>_assignment2_bpe.py --train <train_file.txt> --input <input_file.txt> --vocab_size <size>
