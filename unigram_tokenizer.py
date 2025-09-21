import argparse
import os
import math
from collections import defaultdict
import heapq
import time


LOG_ZERO = -1e18
SPACE_TOKEN = '_'

class TrieNode:
    # A node in the Trie for storing tokens
    def __init__(self):
        self.children = {}
        self.log_prob = None # If not None, this node marks the end of a token

class Trie:
    # Trie data structure for efficient token lookups
    def __init__(self):
        self.root = TrieNode()

    def insert(self, token, log_prob):
        node = self.root
        for char in token:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.log_prob = log_prob

def build_trie_from_log_probs(log_probs):
    # Builds a Trie from a dictionary of token probabilities
    trie = Trie()
    for token, log_prob in log_probs.items():
        trie.insert(token, log_prob)
    return trie


def load_training_data(train_path):
    # Loads training data from a specified file or all files in a directory.
    # It also normalizes whitespace by replacing it with the special SPACE_TOKEN.
    
    text = ""
    if os.path.isdir(train_path):
        print(f"Loading data from directory: {train_path}")
        for filename in sorted(os.listdir(train_path)):
            file_path = os.path.join(train_path, filename)
            if os.path.isfile(file_path):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text += f.read()
    elif os.path.isfile(train_path):
        print(f"Loading data from file: {train_path}")
        with open(train_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
    else:
        raise FileNotFoundError(f"Training path '{train_path}' not found.")
    
    normalized_text = ' '.join(text.split())
    normalized_text = normalized_text.replace(' ', SPACE_TOKEN)
    return normalized_text

def viterbi_segment_trie(text, trie):
    # it finds the optimal segmentation using Viterbi, optimised using Trie.
    # This is used for the E-Step on the large corpus.
    n = len(text)
    dp = [LOG_ZERO] * (n + 1)
    backpointers = [None] * (n + 1)
    dp[0] = 0

    for i in range(n):
        if dp[i] == LOG_ZERO:
            continue
        
        # From start position i, find all possible tokens using the trie
        node = trie.root
        for j in range(i, n):
            char = text[j]
            if char not in node.children:
                break # No more tokens can be formed from this prefix
            node = node.children[char]
            if node.log_prob is not None:
                # We found a valid token: text[i:j+1]
                end_pos = j + 1
                prob = dp[i] + node.log_prob
                if prob > dp[end_pos]:
                    dp[end_pos] = prob
                    backpointers[end_pos] = i
    
    if dp[n] == LOG_ZERO:
        return [], LOG_ZERO
    
    #  Append then reverse is faster than prepending
    tokens = []
    i = n
    while i > 0:
        j = backpointers[i]
        if j is None: # Fallback for characters not in vocab
            tokens.append(text[i-1:i])
            i -= 1
        else:
            tokens.append(text[j:i])
            i = j
    tokens.reverse()
    return tokens, dp[n]

def viterbi_segment_dict(text, log_probs):

    n = len(text)
    dp = [LOG_ZERO] * (n + 1)
    backpointers = [None] * (n + 1)
    dp[0] = 0

    for i in range(1, n + 1):
        for j in range(max(0, i - 12), i): # Max token length here
            token = text[j:i]
            logp = log_probs.get(token)
            if logp is not None:
                prob = dp[j] + logp
                if prob > dp[i]:
                    dp[i] = prob
                    backpointers[i] = j
    
    if dp[n] == LOG_ZERO:
        return [], LOG_ZERO

    tokens = []
    i = n
    while i > 0:
        j = backpointers[i]
        if j is None:
            if i > 0:
                tokens.append(text[i-1:i])
            i -= 1
        else:
            tokens.append(text[j:i])
            i = j
    tokens.reverse()
    return tokens, dp[n]


def train_unigram_tokenizer(text, vocab_size):

    # Initializes a vocabulary and iteratively prunes it down to the target size
    # based on likelihood loss in an EM framework.
    
    # 1. Initialization 
    seed_vocab_size = 20000
    char_counts = defaultdict(int)
    for char in text:
        char_counts[char] += 1
    
    substring_counts = defaultdict(int)
    for i in range(len(text)):
        for j in range(i + 2, min(i + 10, len(text) + 1)):
            substring_counts[text[i:j]] += 1
    
    essential_tokens = set(char_counts.keys())
    seed_vocab = set(essential_tokens)

    frequent_substrings = [sub for sub, count in substring_counts.items() if count >= 2]
    frequent_substrings.sort(key=substring_counts.get, reverse=True)
    
    for sub in frequent_substrings:
        if len(seed_vocab) >= seed_vocab_size:
            break
        seed_vocab.add(sub)
    
    total_initial_counts = float(sum(char_counts.values()) + sum(substring_counts.values()))
    log_probs = {
        token: math.log(char_counts.get(token, substring_counts.get(token, 1.0))) - math.log(total_initial_counts)
        for token in seed_vocab
    }

    # The E-Step optimisation
    while True:
        current_size = len(log_probs)
        if current_size <= vocab_size:
            break
        
        print(f"\n--- Running E-M cycle with vocab size: {current_size} ---")

        # Build Trie for fast E-Step
        model_trie = build_trie_from_log_probs(log_probs)

        # E-Step: Segment the corpus using the fast Trie-based Viterbi
        token_counts = defaultdict(int)
        total_token_count = 0
        chunk_size = 100000
        for chunk_idx in range(0, len(text), chunk_size):
            chunk = text[chunk_idx : chunk_idx + chunk_size]
            tokens, _ = viterbi_segment_trie(chunk, model_trie)
            for token in tokens:
                token_counts[token] += 1
            total_token_count += len(tokens)

        # M-Step: Recalculate log probabilities
        if total_token_count == 0:
            print("Warning: No tokens found in E-step. Stopping.")
            break
        
        # Pre-calculate log total
        log_total_token_count = math.log(total_token_count)
        for token in log_probs:
            log_probs[token] = math.log(token_counts.get(token, 1.0)) - log_total_token_count

        # Loss Calculation: Determine which tokens are least useful
        losses = []
        tokens_to_check = [t for t in log_probs if t not in essential_tokens]
        for token in tokens_to_check:
            original_prob = log_probs.pop(token)
            # Use the simpler dict-based Viterbi for this part
            _, alt_log_prob = viterbi_segment_dict(token, log_probs)
            log_probs[token] = original_prob
            loss = original_prob - alt_log_prob
            losses.append((loss, token))


        num_to_prune_target = int((current_size - vocab_size) * 0.5)
        # ensure we prune a meaningful amount 
        min_prune_amount = int(len(losses) * 0.10)
        num_to_prune = max(num_to_prune_target, min_prune_amount)
        
        # Ensure we don't prune more than needed
        if current_size - num_to_prune < vocab_size:
            num_to_prune = current_size - vocab_size
        
        if num_to_prune <= 0:
            break

        tokens_to_prune = heapq.nsmallest(num_to_prune, losses, key=lambda x: x[0])
        print(f"Pruning... Removing {num_to_prune} token(s).")
        for _, token_to_remove in tokens_to_prune:
            if token_to_remove in log_probs:
                del log_probs[token_to_remove]
    
    final_vocab = sorted(log_probs.keys())
    return final_vocab, log_probs


def save_vocab(vocab, rollno, vocab_size):
    fname = f"{rollno}_assignment2_unigram_vocab_{vocab_size}.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for token in vocab:
            f.write(token + "\n")

def tokenize(text, log_probs):
    # Tokenizes text using the final trained model
    normalized_text = ' '.join(text.split())
    normalized_text = normalized_text.replace(' ', SPACE_TOKEN)
    # For the final tokenization, we can build a Trie for max speed
    model_trie = build_trie_from_log_probs(log_probs)
    tokens, _ = viterbi_segment_trie(normalized_text, model_trie)
    return tokens

def detokenize(tokens):
    return "".join(tokens).replace(SPACE_TOKEN, ' ').strip()

def save_tokens(tokens, rollno):
    fname = f"{rollno}_assignment2_unigram_tokens.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for tok in tokens:
            f.write(tok + "\n")

def save_detokenized(text, rollno):
    fname = f"{rollno}_assignment2_unigram_detokenized.txt"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimized Unigram Tokenizer Training")
    parser.add_argument("--train", type=str, required=True, help="Path to the training data file or directory.")
    parser.add_argument("--input", type=str, required=True, help="Path to the file to tokenize.")
    parser.add_argument("--vocab_size", type=int, required=True, help="Desired final vocabulary size.")
    args = parser.parse_args()

    rollno = "220568"

    print("--- Step 1: Loading and Preparing Training Data ---")
    train_text = load_training_data(args.train)
    print(f"Training data loaded ({len(train_text):,} characters).\n")
    
    print("--- Step 2: Training Unigram Tokenizer ---")
    training_start_time = time.perf_counter()
    
    vocab, tokenizer_model = train_unigram_tokenizer(train_text, args.vocab_size)
    
    training_end_time = time.perf_counter()
    duration_seconds = training_end_time - training_start_time
    
    print(f"\nTraining complete. Final vocabulary size: {len(vocab)}")
    print(f"Training took {duration_seconds:.2f} seconds ({duration_seconds/60:.2f} minutes).")
    
    print("\n--- Step 3: Saving Vocabulary ---")
    save_vocab(vocab, rollno, args.vocab_size)
    print(f"Vocabulary saved to {rollno}_assignment2_unigram_vocab_{args.vocab_size}.txt")

    print("\n--- Step 4: Tokenizing Sample Text ---")
    with open(args.input, "r", encoding="utf-8") as f:
        sample_text = f.read()
    print("Sample text loaded.")
    
    tokens = tokenize(sample_text, tokenizer_model)
    save_tokens(tokens, rollno)
    print(f"Tokens saved to {rollno}_assignment2_unigram_tokens.txt")
    
    print("\n--- Step 5: Detokenizing Text ---")
    detok_text = detokenize(tokens)
    save_detokenized(detok_text, rollno)
    print(f"Detokenized text saved to {rollno}_assignment2_unigram_detokenized.txt")
    
    print("\n--- Process Finished ---")
    # print("\n--- Sample of Results ---")
    # print(f"Original Text (first 200 chars):    '{sample_text[:200]}...'")
    # print(f"Tokenized Representation (first 20): {tokens[:20]}")
    # print(f"Detokenized Text (first 200 chars): '{detok_text[:200]}...'")
