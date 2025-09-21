import argparse
import math
import os
import sys
import unicodedata
from collections import Counter, defaultdict
import heapq


CONTINUATION_PREFIX = "##"
SPECIAL_TOKENS = ["<pad>", "<unk>", "<s>", "</s>"]

def normalize_string(text):
    # Applies NFKC normalization to a string.
    return unicodedata.normalize("NFKC", text)

def tokenize_by_space(text_line):
    # Splits a line of text by whitespace.
    return text_line.strip().split()


class VocabTrieNode:
    # A node in the vocabulary Trie
    __slots__ = ("children", "is_token_end")
    def __init__(self):
        self.children = {}
        self.is_token_end = False

class VocabTrie:
    # A Trie data structure to efficiently find the longest matching token
    def __init__(self):
        self.root = VocabTrieNode()

    def add_token(self, token):
        # Adds a token string to the Trie.
        node = self.root
        for char in token:
            if char not in node.children:
                node.children[char] = VocabTrieNode()
            node = node.children[char]
        node.is_token_end = True

    def find_longest_match(self, text, start_pos=0):
        # Finds the longest token in the Trie that is a prefix of the text
        node = self.root
        longest_match_len = 0
        current_len = 0
        i = start_pos
        while i < len(text):
            char = text[i]
            if char in node.children:
                node = node.children[char]
                current_len += 1
                if node.is_token_end:
                    longest_match_len = current_len
                i += 1
            else:
                break
        
        if longest_match_len == 0:
            return 0, None
        
        return longest_match_len, text[start_pos : start_pos + longest_match_len]


class WordPieceTrainer:
    
    # Implements the WordPiece tokenization training algorithm using a node pool

    def __init__(self, vocab_target, is_verbose=False):
        self.vocab_target = int(vocab_target)
        self.is_verbose = is_verbose

        # Vocabulary mapping
        self.id_to_token = []
        self.token_to_id = {}
        for token in SPECIAL_TOKENS:
            self._register_token(token)

        # Corpus word data
        self.word_to_wid = {}
        self.word_counts = []

        # Node pool represented as a doubly-linked list for each word
        self.nodes_token_id = []
        self.nodes_next = []
        self.nodes_prev = []
        self.nodes_word_id = []
        self.nodes_is_dead = []

        # Word-level metadata for the node pool
        self.word_start_node = []

        # Data structures for tracking pairs and merges
        self.pair_frequencies = Counter()
        self.pair_locations = defaultdict(list)
        self.token_graph = defaultdict(set)
        self.merge_queue = []
        self.merge_history = []
        
        # Total number of tokens in the corpus representation
        self.total_token_count = 0
        
        # Set of tokens that can start a word
        self.start_tokens = set()
        
        # Trie for final tokenization
        self.tokenizer_trie = None

    def _register_token(self, token_str):
        # Adds a new token to the vocabulary if it doesn't exist 
        if token_str in self.token_to_id:
            return self.token_to_id[token_str]
        
        new_id = len(self.id_to_token)
        self.id_to_token.append(token_str)
        self.token_to_id[token_str] = new_id
        return new_id

    def load_and_prep_corpus(self, corpus_path):
        # Reads the training corpus, counts words, and builds the initial node pool
        word_freq_counter = Counter()
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                normalized_line = normalize_string(line)
                if not normalized_line:
                    continue
                for word in tokenize_by_space(normalized_line):
                    word_freq_counter[word] += 1

        for word, count in sorted(word_freq_counter.items()):
            wid = len(self.word_counts)
            self.word_to_wid[word] = wid
            self.word_counts.append(count)
            
            head_node_idx = -1
            last_node_idx = -1
            word_node_indices = []

            # Create a doubly-linked list of character nodes for the word
            for char_pos, char in enumerate(word):
                tid = self._register_token(char)
                node_idx = len(self.nodes_token_id)
                
                self.nodes_token_id.append(tid)
                self.nodes_next.append(-1)
                self.nodes_prev.append(last_node_idx)
                self.nodes_word_id.append(wid)
                self.nodes_is_dead.append(False)

                if last_node_idx != -1:
                    self.nodes_next[last_node_idx] = node_idx
                else:
                    head_node_idx = node_idx
                
                last_node_idx = node_idx
                word_node_indices.append(node_idx)

            self.word_start_node.append(head_node_idx)
            self.total_token_count += len(word_node_indices) * count

            # Initialize pair counts from the new word
            for i in range(len(word_node_indices) - 1):
                left_node = word_node_indices[i]
                right_node = word_node_indices[i+1]
                left_tid = self.nodes_token_id[left_node]
                right_tid = self.nodes_token_id[right_node]
                
                self.pair_frequencies[(left_tid, right_tid)] += count
                self.pair_locations[(left_tid, right_tid)].append(left_node)
                self.token_graph[left_tid].add(right_tid)
                self.token_graph[right_tid].add(left_tid)
        
        if self.is_verbose:
            print(f"Initialization complete. Words: {len(self.word_counts)}, Tokens: {len(self.id_to_token)}, N: {self.total_token_count}", file=sys.stderr)

    def _calculate_merge_score(self, pair):
        # Calculates the score for a potential merge, based on log-likelihood
        freq = self.pair_frequencies.get(pair, 0)
        if freq <= 0 or self.total_token_count <= 0:
            return float("-inf")
        # The score is derived from the change in corpus log-likelihood
        return freq * (math.log(self.total_token_count) - math.log(freq))

    def _add_pair_to_queue(self, left_tid, right_tid):
        # Calculates score for a pair and pushes it to the priority queue
        pair = (left_tid, right_tid)
        if self.pair_frequencies.get(pair, 0) <= 0:
            return
            
        score = self._calculate_merge_score(pair)
        left_str = self.id_to_token[left_tid]
        right_str = self.id_to_token[right_tid]
        # Use negative score because heapq is a min-heap
        heapq.heappush(self.merge_queue, (-score, left_str, right_str, left_tid, right_tid))

    def _populate_merge_queue(self):
        # Initializes the priority queue with all existing pairs
        self.merge_queue = []
        # Sorting ensures deterministic behavior for pairs with the same score
        sorted_pairs = sorted(list(self.pair_frequencies.keys()), key=lambda p: (self.id_to_token[p[0]], self.id_to_token[p[1]]))
        for left_tid, right_tid in sorted_pairs:
            self._add_pair_to_queue(left_tid, right_tid)

    def run_training_merges(self):
        # Performs iterative merging of token pairs until the target vocab size is reached
        initial_vocab_size = len(self.id_to_token)
        num_merges_to_perform = self.vocab_target - initial_vocab_size

        if num_merges_to_perform <= 0:
            if self.is_verbose:
                print("Target vocab size is not greater than initial size. No merges needed.", file=sys.stderr)
            return

        if self.is_verbose:
            print(f"Target: {self.vocab_target}, Initial: {initial_vocab_size}, Merges needed: {num_merges_to_perform}", file=sys.stderr)

        self._populate_merge_queue()
        merges_completed = 0
        
        while merges_completed < num_merges_to_perform and self.merge_queue:
            neg_score, _, _, left_tid, right_tid = heapq.heappop(self.merge_queue)
            
            pair_to_merge = (left_tid, right_tid)
            
            # Skip if this pair has been processed or its frequency dropped to zero
            if self.pair_frequencies.get(pair_to_merge, 0) <= 0:
                continue

            # Stale entry check: re-calculate score and push back if it changed
            current_score = self._calculate_merge_score(pair_to_merge)
            popped_score = -neg_score
            if abs(popped_score - current_score) > 1e-11:
                heapq.heappush(self.merge_queue, (-current_score, self.id_to_token[left_tid], self.id_to_token[right_tid], left_tid, right_tid))
                continue

            # --- Perform the merge ---
            new_token_str = self.id_to_token[left_tid] + self.id_to_token[right_tid]
            new_tid = self._register_token(new_token_str)
            self.merge_history.append(new_token_str)
            merges_completed += 1

            locations = self.pair_locations.pop(pair_to_merge, [])
            if not locations:
                continue
            
            # Process all occurrences of this pair in the corpus
            for left_node in sorted(locations):
                # Check for validity of the node and its pair
                if self.nodes_is_dead[left_node] or self.nodes_token_id[left_node] != left_tid:
                    continue
                right_node = self.nodes_next[left_node]
                if right_node == -1 or self.nodes_is_dead[right_node] or self.nodes_token_id[right_node] != right_tid:
                    continue
                
                word_id = self.nodes_word_id[left_node]
                freq = self.word_counts[word_id]
                
                prev_node = self.nodes_prev[left_node]
                next_node = self.nodes_next[right_node]

                # Update counts for pairs affected by this merge
                self.pair_frequencies[pair_to_merge] -= freq
                if prev_node != -1:
                    self.pair_frequencies[(self.nodes_token_id[prev_node], left_tid)] -= freq
                if next_node != -1:
                    self.pair_frequencies[(right_tid, self.nodes_token_id[next_node])] -= freq

                # Update the node list
                self.nodes_token_id[left_node] = new_tid
                self.nodes_is_dead[right_node] = True
                self.nodes_next[left_node] = next_node
                if next_node != -1:
                    self.nodes_prev[next_node] = left_node
                
                # One merge reduces the total token count by one
                self.total_token_count -= freq

                # Create new pairs and update their counts
                if prev_node != -1:
                    prev_tid = self.nodes_token_id[prev_node]
                    new_pair = (prev_tid, new_tid)
                    self.pair_frequencies[new_pair] += freq
                    self.pair_locations[new_pair].append(prev_node)
                    self.token_graph[prev_tid].add(new_tid)
                    self.token_graph[new_tid].add(prev_tid)
                
                if next_node != -1:
                    next_tid = self.nodes_token_id[next_node]
                    new_pair = (new_tid, next_tid)
                    self.pair_frequencies[new_pair] += freq
                    self.pair_locations[new_pair].append(left_node)
                    self.token_graph[new_tid].add(next_tid)
                    self.token_graph[next_tid].add(new_tid)

            # Add newly formed or affected pairs back to the queue
            affected_neighbors = set()
            affected_neighbors.update(self.token_graph.get(new_tid, set()))
            affected_neighbors.update(self.token_graph.get(left_tid, set()))
            affected_neighbors.update(self.token_graph.get(right_tid, set()))
            
            potential_new_pairs = []
            for neighbor_tid in affected_neighbors:
                if self.pair_frequencies.get((new_tid, neighbor_tid), 0) > 0:
                    potential_new_pairs.append((new_tid, neighbor_tid))
                if self.pair_frequencies.get((neighbor_tid, new_tid), 0) > 0:
                    potential_new_pairs.append((neighbor_tid, new_tid))
            
            for l, r in sorted(potential_new_pairs, key=lambda p: (self.id_to_token[p[0]], self.id_to_token[p[1]])):
                self._add_pair_to_queue(l, r)

        if self.is_verbose:
            print(f"Training finished. Merges: {merges_completed}, Final vocab: {len(self.id_to_token)}", file=sys.stderr)

    def _construct_tokenizer_trie(self):
        # Builds a Trie from the final vocabulary for fast tokenization
        self.tokenizer_trie = VocabTrie()
        for token in self.id_to_token:
            self.tokenizer_trie.add_token(token)

    def _identify_start_tokens(self, corpus_path):
        # Determines which vocabulary tokens can appear at the start of a word
        start_token_candidates = set()
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                normalized_line = normalize_string(line)
                if not normalized_line:
                    continue
                for word in tokenize_by_space(normalized_line):
                    if not word:
                        continue
                    
                    match_len, token = self.tokenizer_trie.find_longest_match(word, 0)
                    if match_len > 0:
                        start_token_candidates.add(token)
        self.start_tokens = start_token_candidates

    def save_vocabulary(self, output_path):
        # Exports the final vocabulary to a file
        final_vocab = []
        final_vocab.extend(SPECIAL_TOKENS)
        
        # Add merged tokens in the order they were created
        for token in self.merge_history:
            if len(final_vocab) >= self.vocab_target:
                break
            final_vocab.append(token)
        
        # Fill remaining slots with base character tokens
        if len(final_vocab) < self.vocab_target:
            existing_tokens = set(final_vocab)
            remaining_chars = []
            for token in self.id_to_token:
                if token not in existing_tokens:
                    remaining_chars.append(token)
            
            for token in sorted(remaining_chars):
                if len(final_vocab) >= self.vocab_target:
                    break
                final_vocab.append(token)

        with open(output_path, "w", encoding="utf-8") as f:
            for token in final_vocab[:self.vocab_target]:
                if token in SPECIAL_TOKENS:
                    f.write(token + "\n")
                else:
                    # Non-start tokens get the continuation prefix
                    prefix = "" if token in self.start_tokens else CONTINUATION_PREFIX
                    f.write(prefix + token + "\n")

    def encode(self, text_line):
        # Tokenizes a single line of text into WordPieces
        output_tokens = []
        for word in tokenize_by_space(text_line):
            if not word:
                continue

            pos = 0
            num_chars = len(word)
            word_pieces = []
            is_first_piece = True
            
            while pos < num_chars:
                match_len, token = self.tokenizer_trie.find_longest_match(word, pos)
                if match_len == 0:
                    # Cannot segment the word, mark as unknown
                    word_pieces = ["<unk>"]
                    break
                
                piece = token if is_first_piece else CONTINUATION_PREFIX + token
                word_pieces.append(piece)
                pos += match_len
                is_first_piece = False

            output_tokens.extend(word_pieces)
        return output_tokens
    
    def decode(self, token_sequence):
        # Converts a sequence of tokens back into a string
        words = []
        current_word = ""
        for token in token_sequence:
            if token == "<unk>":
                if current_word:
                    words.append(current_word)
                words.append(token)
                current_word = ""
            elif token.startswith(CONTINUATION_PREFIX):
                current_word += token[len(CONTINUATION_PREFIX):]
            else:
                if current_word:
                    words.append(current_word)
                current_word = token
        
        if current_word:
            words.append(current_word)
            
        return " ".join(words)

def main():
    # Main function to handle argument parsing and execution flow
    parser = argparse.ArgumentParser(description="WordPiece Tokenizer Trainer")
    parser.add_argument("--train", required=True, help="Path to the training corpus file.")
    parser.add_argument("--input", required=True, help="Path to the input file to tokenize.")
    parser.add_argument("--vocab_size", required=True, type=int, help="Target vocabulary size.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output during training.")
    # The --no-progress argument is kept for compatibility but does nothing.
    parser.add_argument("--no-progress", action="store_true", help="This argument is ignored.")
    
    config = parser.parse_args()
    
    ROLL_NUMBER = 220568
    vocab_output_file = f"{ROLL_NUMBER}_assignment2_wp_vocab_{config.vocab_size}.txt"
    tokens_output_file = f"{ROLL_NUMBER}_assignment2_wp_tokens.txt"
    detokenized_output_file = f"{ROLL_NUMBER}_assignment2_wp_detokenized.txt"

    trainer_instance = WordPieceTrainer(vocab_target=config.vocab_size, is_verbose=config.verbose)
    
    print("Reading training corpus and initializing...", file=sys.stderr)
    trainer_instance.load_and_prep_corpus(config.train)
    
    print("Starting training (merging pairs)...", file=sys.stderr)
    trainer_instance.run_training_merges()

    # --- Vocabulary and Tokenizer Preparation ---
    print("Building Trie for tokenization...", file=sys.stderr)
    trainer_instance._construct_tokenizer_trie()

    print("Identifying start-tokens for vocab export...", file=sys.stderr)
    trainer_instance._identify_start_tokens(config.train)
    
    print(f"Exporting vocabulary to {vocab_output_file}...", file=sys.stderr)
    trainer_instance.save_vocabulary(vocab_output_file)

    # --- Tokenization and Detokenization ---
    print(f"Tokenizing {config.input}...", file=sys.stderr)
    detokenized_lines = []
    with open(config.input, "r", encoding="utf-8") as text_in, \
         open(tokens_output_file, "w", encoding="utf-8") as tokens_out:
        for line in text_in:
            original_line = line.rstrip("\n")
            normalized_line = normalize_string(original_line)
            tokens = trainer_instance.encode(normalized_line)
            
            for t in tokens:
                tokens_out.write(t + "\n")
            
            detokenized_line = trainer_instance.decode(tokens)
            detokenized_lines.append(detokenized_line)
    
    print(f"Writing detokenized output to {detokenized_output_file}...", file=sys.stderr)
    with open(detokenized_output_file, "w", encoding="utf-8") as detok_out:
        for i, line in enumerate(detokenized_lines):
            detok_out.write(line)
            if i < len(detokenized_lines) - 1:
                detok_out.write("\n")

    print("\nProcessing complete.", file=sys.stderr)
    print(f"  Vocabulary: {vocab_output_file}", file=sys.stderr)
    print(f"  Tokens: {tokens_output_file}", file=sys.stderr)
    print(f"  Detokenized: {detokenized_output_file}", file=sys.stderr)

if __name__ == "__main__":
    main()