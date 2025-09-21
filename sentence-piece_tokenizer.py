import argparse
import os
import collections
import time
import heapq
import unicodedata



# Creates a mapping from all 256 possible byte values to a unique Unicode character.
def generate_byte_char_map():
    return {i: chr(i) for i in range(256)}

# Global maps and constants
BYTE_TO_CHAR_MAP = generate_byte_char_map()
CHAR_TO_BYTE_MAP = {char: byte for byte, char in BYTE_TO_CHAR_MAP.items()}
SPACE_MARKER = " " # Special character to represent spaces

# A node in a doubly linked list representing a single token.
class SPMNode:
    # Using __slots__ for memory optimization on large corpora
    __slots__ = ['token_id', 'chunk_id', 'index', 'prev_node', 'next_node', 'is_active']
    def __init__(self, token_id, chunk_id, index):
        self.token_id = token_id
        self.chunk_id = chunk_id
        self.index = index
        self.prev_node = None
        self.next_node = None
        self.is_active = True

# Manages a sequence of tokens as a linked list for a chunk of text.
class SPMSequence:
    def __init__(self, token_ids, chunk_id):
        self.chunk_id = chunk_id
        self.start_node = None
        
        # Build linked list from token IDs
        if not token_ids:
            return
        
        nodes_list = [SPMNode(tok_id, chunk_id, i) for i, tok_id in enumerate(token_ids)]
        i = 0
        while i < len(nodes_list):
            if i > 0:
                nodes_list[i].prev_node = nodes_list[i-1]
            if i < len(nodes_list) - 1:
                nodes_list[i].next_node = nodes_list[i+1]
            i += 1
        self.start_node = nodes_list[0]
    
    # Merges two adjacent nodes in O(1) time.
    def perform_merge(self, node_one, node_two, new_token_id):
        # Ensure nodes are valid and adjacent
        if not (node_one.is_active and node_two.is_active and node_one.next_node == node_two):
            return None
        
        merged_node = SPMNode(new_token_id, self.chunk_id, node_one.index)
        
        # Rewire pointers
        merged_node.prev_node = node_one.prev_node
        merged_node.next_node = node_two.next_node
        
        if merged_node.prev_node:
            merged_node.prev_node.next_node = merged_node
        else:
            self.start_node = merged_node
            
        if merged_node.next_node:
            merged_node.next_node.prev_node = merged_node
        
        # Deactivate old nodes
        node_one.is_active = False
        node_two.is_active = False
        
        return merged_node
    
    # Retrieves all active adjacent token pairs.
    def get_current_pairs(self):
        pairs = []
        node = self.start_node
        while node and node.next_node:
            if node.is_active and node.next_node.is_active:
                pairs.append((node, node.next_node))
            node = node.next_node
        return pairs

# Tracks pair frequencies and manages the priority queue.
class SPMFreqTracker:
    def __init__(self):
        self.pair_locations = collections.defaultdict(list)
        self.pair_frequencies = collections.defaultdict(int)
        self.priority_queue = []
        self.timestamp_counter = 0
        self.pair_timestamps = {}
        self.chunk_map = {} # Maps chunk_id to SPMSequence object
    
    # Adds a text chunk (sequence) to the tracker.
    def add_chunk(self, sequence):
        self.chunk_map[sequence.chunk_id] = sequence
        for start_node, _ in sequence.get_current_pairs():
            self.add_occurrence(start_node)
    
    # Records a new pair occurrence.
    def add_occurrence(self, start_node):
        if start_node.next_node is None: return
        pair = (start_node.token_id, start_node.next_node.token_id)
        self.pair_locations[pair].append(start_node)
        self.pair_frequencies[pair] += 1
    
    # Reduces the count of a pair.
    def remove_occurrence(self, start_node):
        if start_node.next_node is None: return
        pair = (start_node.token_id, start_node.next_node.token_id)
        self.pair_frequencies[pair] = max(0, self.pair_frequencies[pair] - 1)
    
    # Builds the initial heap.
    def create_heap(self):
        self.priority_queue = []
        for pair, freq in self.pair_frequencies.items():
            if freq > 0:
                self.timestamp_counter += 1
                self.pair_timestamps[pair] = self.timestamp_counter
                heapq.heappush(self.priority_queue, (-freq, pair, self.timestamp_counter))

    # Updates a pair's entry in the heap.
    def update_heap(self, pair):
        freq = self.pair_frequencies.get(pair, 0)
        if freq > 0:
            self.timestamp_counter += 1
            self.pair_timestamps[pair] = self.timestamp_counter
            heapq.heappush(self.priority_queue, (-freq, pair, self.timestamp_counter))

    # Gets the highest-priority valid pair.
    def get_top_pair(self):
        while self.priority_queue:
            inv_freq, pair, ts_val = heapq.heappop(self.priority_queue)
            if self.pair_timestamps.get(pair) != ts_val: continue
            if self.pair_frequencies.get(pair, 0) != -inv_freq: continue
            return pair
        return None
    
    # Gets all valid locations of a pair.
    def get_valid_locations(self, pair):
        if pair not in self.pair_locations: return []
        
        active_locations = [
            node for node in self.pair_locations[pair]
            if (node.is_active and node.next_node and node.next_node.is_active and 
                node.token_id == pair[0] and node.next_node.token_id == pair[1])
        ]
        
        self.pair_locations[pair] = active_locations
        self.pair_frequencies[pair] = len(active_locations)
        return active_locations

# Wrapper for the trained SPM-style model.
class SPMModel:
    def __init__(self, id_to_token_map):
        self.id_to_token_map = id_to_token_map
        self.token_to_id_map = {tok: i for i, tok in self.id_to_token_map.items()}
        
        # Build a trie for efficient longest-match search
        self.token_trie = {}
        for token_str, token_id in self.token_to_id_map.items():
            trie_level = self.token_trie
            for char in token_str:
                if char not in trie_level:
                    trie_level[char] = {}
                trie_level = trie_level[char]
            trie_level['#'] = token_id # Mark end of token


def normalize_and_prepare_text(text):
    text = unicodedata.normalize('NFKC', text)
    text = text.lower()
    text = text.replace(' ', SPACE_MARKER)
    if not text.startswith(SPACE_MARKER):
        text = SPACE_MARKER + text
    return text


def train_bpe_tokenizer(text, vocab_size):
    processed_text = normalize_and_prepare_text(text)
    
    # 1. Initialize vocabulary
    special_tokens = ["<pad>", "<unk>", "<s>", "</s>"]
    token_lookup = {BYTE_TO_CHAR_MAP[i]: i for i in range(256)}
    for tok in special_tokens:
        if tok not in token_lookup:
            token_lookup[tok] = len(token_lookup)
    
    id_lookup = {i: tok for tok, i in token_lookup.items()}
    
    # 2. Process text in chunks to manage memory
    tracker = SPMFreqTracker()
    byte_representation = [token_lookup[BYTE_TO_CHAR_MAP[b]] for b in processed_text.encode('utf-8')]
    
    # Process 1MB chunks
    chunk_size = 1 * 1024 * 1024
    chunk_start_idx = 0
    chunk_id = 0
    while chunk_start_idx < len(byte_representation):
        chunk_end_idx = min(chunk_start_idx + chunk_size, len(byte_representation))
        chunk_data = byte_representation[chunk_start_idx:chunk_end_idx]
        
        sequence = SPMSequence(chunk_data, chunk_id)
        tracker.add_chunk(sequence)
        
        chunk_start_idx = chunk_end_idx
        chunk_id += 1
    
    # 3. Build priority queue
    tracker.create_heap()
    
    # 4. Main merge loop
    merge_rules = []
    num_merges_needed = vocab_size - len(token_lookup)
    
    for i in range(num_merges_needed):
        top_pair = tracker.get_top_pair()
        if top_pair is None:
            print(f"Stopping early: No more pairs to merge after {i} merges.")
            break
            
        # Add new token
        new_id = len(token_lookup)
        tok1, tok2 = top_pair
        new_token_string = id_lookup[tok1] + id_lookup[tok2]
        
        token_lookup[new_token_string] = new_id
        id_lookup[new_id] = new_token_string
        merge_rules.append(top_pair)
        
        # Apply merge
        locations_to_merge = tracker.get_valid_locations(top_pair)
        pairs_to_update = set()
        
        for start_node in locations_to_merge:
            end_node = start_node.next_node
            if not end_node: continue
            
            # Decrement counts of affected pairs
            if start_node.prev_node:
                tracker.remove_occurrence(start_node.prev_node)
                pairs_to_update.add((start_node.prev_node.token_id, start_node.token_id))
            tracker.remove_occurrence(start_node)
            if end_node.next_node:
                tracker.remove_occurrence(end_node)
                pairs_to_update.add((end_node.token_id, end_node.next_node.token_id))
            
            # Perform merge and add new pairs
            sequence = tracker.chunk_map[start_node.chunk_id]
            new_node = sequence.perform_merge(start_node, end_node, new_id)
            
            if new_node:
                if new_node.prev_node:
                    tracker.add_occurrence(new_node.prev_node)
                    pairs_to_update.add((new_node.prev_node.token_id, new_node.token_id))
                if new_node.next_node:
                    tracker.add_occurrence(new_node)
                    pairs_to_update.add((new_node.token_id, new_node.next_node.token_id))
        
        for pair in pairs_to_update:
            tracker.update_heap(pair)

    # 5. Finalize vocabulary
    final_vocab = list(special_tokens)
    final_vocab.extend([BYTE_TO_CHAR_MAP[i] for i in range(256) if BYTE_TO_CHAR_MAP[i] not in special_tokens])
    final_vocab.extend([id_lookup[p1] + id_lookup[p2] for p1, p2 in merge_rules])
    
    tokenizer_model = SPMModel(id_lookup)
    return final_vocab, tokenizer_model

def tokenize(text, tokenizer):
    # Use longest-match search with a trie
    processed_text = normalize_and_prepare_text(text)
    char_sequence = [BYTE_TO_CHAR_MAP[b] for b in processed_text.encode('utf-8')]

    output_tokens = []
    idx = 0
    while idx < len(char_sequence):
        longest_match_len = 0
        longest_match_id = -1
        
        trie_level = tokenizer.token_trie
        
        # Search for the longest possible token starting at idx
        search_idx = idx
        while search_idx < len(char_sequence):
            char = char_sequence[search_idx]
            if char in trie_level:
                trie_level = trie_level[char]
                if '#' in trie_level: # Found a valid token
                    longest_match_len = search_idx - idx + 1
                    longest_match_id = trie_level['#']
            else:
                break
            search_idx += 1
        
        if longest_match_len == 0:
            # If no match, add the single character and advance
            output_tokens.append(char_sequence[idx])
            idx += 1
        else:
            # Add the longest found token and jump ahead
            output_tokens.append(tokenizer.id_to_token_map[longest_match_id])
            idx += longest_match_len
            
    return output_tokens

def detokenize(tokens, tokenizer):
    # Join all tokens and decode from byte representation
    full_string = "".join(tokens)
    buffer = bytearray()
    
    for char in full_string:
        # Special tokens are ignored during detokenization
        if char in CHAR_TO_BYTE_MAP:
            buffer.append(CHAR_TO_BYTE_MAP[char])
            
    decoded_text = buffer.decode('utf-8', errors='replace')
    # Restore spaces
    restored_text = decoded_text.replace(SPACE_MARKER, ' ')
    
    return restored_text.strip()

def load_training_data(train_path):
    with open(train_path, "r", encoding="utf-8") as f:
        return f.read()

def save_vocab(vocab, rollno, vocab_size):
    fname = f"{rollno}_assignment2_spm_vocab_{vocab_size}.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for token in vocab:
            f.write(token + "\n")
    print(f"Vocabulary of size {len(vocab)} saved to {fname}")

def save_tokens(tokens, rollno):
    fname = f"{rollno}_assignment2_spm_tokens.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for tok in tokens:
            f.write(tok + "\n")
    print(f"Tokens saved to {fname}")

def save_detokenized(text, rollno):
    fname = f"{rollno}_assignment2_spm_detokenized.txt"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Detokenized text saved to {fname}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    args = parser.parse_args()

    rollno = "220568"

    # Training
    print("Loading training data...")
    train_text = load_training_data(args.train)
    print(f"Training SPM-style BPE tokenizer for vocab size {args.vocab_size}...")
    vocab, tokenizer = train_bpe_tokenizer(train_text, args.vocab_size)
    save_vocab(vocab, rollno, args.vocab_size)

    # Tokenization
    print("\nTokenizing input file...")
    with open(args.input, "r", encoding="utf-8") as f:
        sample_text = f.read()
    tokens = tokenize(sample_text, tokenizer)
    save_tokens(tokens, rollno)

    # Detokenization
    print("\nDetokenizing...")
    detok_text = detokenize(tokens, tokenizer)
    save_detokenized(detok_text, rollno)