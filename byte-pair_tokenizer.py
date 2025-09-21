import argparse
import os
import collections
import time
import heapq


# Creates a mapping from all 256 possible byte values to a unique Unicode character.
def generate_byte_char_map():
    return {i: chr(i) for i in range(256)}

# Global maps for byte-character conversion
BYTE_TO_CHAR_MAP = generate_byte_char_map()
CHAR_TO_BYTE_MAP = {char: byte for byte, char in BYTE_TO_CHAR_MAP.items()}

# A node in a doubly linked list representing a single token.
class ChainNode:
    def __init__(self, token_id, word_ref_id, index):
        self.token_id = token_id
        self.word_ref_id = word_ref_id
        self.index = index
        self.prev_node = None
        self.next_node = None
        self.is_active = True # Used for lazy deletion
        self.parent_chain = None # Direct reference to the parent list

# Manages a sequence of tokens as a linked list.
class TokenChain:
    def __init__(self, token_ids, word_ref_id):
        self.word_ref_id = word_ref_id
        self.start_node = None
        self.nodes = []

        # Construct the linked list from token IDs
        if not token_ids:
            return
            
        for i in range(len(token_ids)):
            node = ChainNode(token_ids[i], word_ref_id, i)
            node.parent_chain = self # Link back to parent
            self.nodes.append(node)
            if i == 0:
                self.start_node = node
            else:
                self.nodes[i-1].next_node = node
                node.prev_node = self.nodes[i-1]
    
    # Merges two adjacent nodes in O(1) time.
    def perform_merge(self, node_one, node_two, new_token_id):
        # Ensure nodes are valid and adjacent before merging
        if not (node_one.is_active and node_two.is_active and node_one.next_node == node_two):
            return None
            
        merged_node = ChainNode(new_token_id, self.word_ref_id, node_one.index)
        merged_node.parent_chain = self
        
        # Rewire the linked list pointers
        merged_node.prev_node = node_one.prev_node
        merged_node.next_node = node_two.next_node
        
        if merged_node.prev_node:
            merged_node.prev_node.next_node = merged_node
        else:
            self.start_node = merged_node # Update list head
            
        if merged_node.next_node:
            merged_node.next_node.prev_node = merged_node
        
        # Deactivate the old nodes instead of deleting
        node_one.is_active = False
        node_two.is_active = False
        
        return merged_node
    
    # Retrieves all active adjacent token pairs from the list.
    def get_current_pairs(self):
        pairs = []
        node = self.start_node
        while node and node.next_node:
            if node.is_active and node.next_node.is_active:
                pairs.append((node, node.next_node))
            node = node.next_node
        return pairs
    
    # Converts the linked list back to a flat list of token IDs.
    def to_id_list(self):
        token_ids = []
        node = self.start_node
        while node:
            if node.is_active:
                token_ids.append(node.token_id)
            node = node.next_node
        return token_ids

# Tracks pair frequencies and manages the priority queue for merges.
class PairManager:
    def __init__(self):
        # (tok1, tok2) -> list of starting ChainNodes
        self.pair_locations = collections.defaultdict(list)
        # (tok1, tok2) -> total count
        self.pair_frequencies = collections.defaultdict(int)
        # Heap for finding the most frequent pair: (-frequency, (tok1, tok2), timestamp)
        self.priority_queue = []
        self.timestamp_counter = 0
        # Stores the latest timestamp for a pair to handle stale entries in the queue
        self.pair_timestamps = {}
    
    # Records a new occurrence of a pair.
    def add_occurrence(self, start_node):
        end_node = start_node.next_node
        if end_node is None:
            return
            
        pair = (start_node.token_id, end_node.token_id)
        self.pair_locations[pair].append(start_node)
        self.pair_frequencies[pair] += 1
    
    # Builds the initial heap from all counted pairs.
    def create_heap(self):
        self.priority_queue = []
        for pair, freq in self.pair_frequencies.items():
            if freq > 0:
                self.timestamp_counter += 1
                self.pair_timestamps[pair] = self.timestamp_counter
                entry = (-freq, pair, self.timestamp_counter)
                heapq.heappush(self.priority_queue, entry)
    
    # Pushes an updated pair frequency to the heap.
    def update_heap(self, pair):
        if pair in self.pair_frequencies and self.pair_frequencies[pair] > 0:
            freq = self.pair_frequencies[pair]
            self.timestamp_counter += 1
            self.pair_timestamps[pair] = self.timestamp_counter
            entry = (-freq, pair, self.timestamp_counter)
            heapq.heappush(self.priority_queue, entry)
    
    # Pops the most frequent, valid pair from the heap.
    def get_top_pair(self):
        while self.priority_queue:
            inv_freq, pair, ts_val = heapq.heappop(self.priority_queue)
            
            # Discard stale entries from the queue
            if self.pair_timestamps.get(pair) != ts_val:
                continue
            
            # Discard if the frequency in the queue doesn't match the current one
            if self.pair_frequencies.get(pair, 0) != -inv_freq:
                continue
                
            return pair
            
        return None
    
    # Cleans and returns all valid locations for a given pair.
    def get_valid_locations(self, pair):
        if pair not in self.pair_locations:
            return []
        
        all_locations = self.pair_locations[pair]
        active_locations = []
        for start_node in all_locations:
            # Check if the node and its successor are still valid and form the correct pair
            is_node_valid = start_node.is_active
            is_next_valid = start_node.next_node and start_node.next_node.is_active
            is_pair_correct = start_node.token_id == pair[0] and start_node.next_node.token_id == pair[1]

            if is_node_valid and is_next_valid and is_pair_correct:
                active_locations.append(start_node)
        
        # Update the stored locations and frequency
        self.pair_locations[pair] = active_locations
        self.pair_frequencies[pair] = len(active_locations)
        return active_locations

# A simple wrapper for the trained BPE model.
class BPEModel:
    def __init__(self, merge_rules, id_to_token_map):
        self.merge_rules = merge_rules
        self.id_to_token_map = id_to_token_map

# --- Core BPE Functions ---

def train_bpe_tokenizer(text, vocab_size):
    # 1. Initialize vocabulary
    special_tokens = ["<pad>", "<unk>", "<s>", "</s>"]
    eow_marker = "</w>"
    
    token_lookup = {BYTE_TO_CHAR_MAP[i]: i for i in range(256)}
    for tok in special_tokens + [eow_marker]:
        if tok not in token_lookup:
            token_lookup[tok] = len(token_lookup)
    
    id_lookup = {i: tok for tok, i in token_lookup.items()}
    
    # 2. Pre-tokenize and count word frequencies
    word_counts = collections.Counter(text.split())
    
    # 3. Initialize data structures
    word_chains = {}
    manager = PairManager()
    
    for word_id, word in enumerate(word_counts):
        # Convert word to byte tokens and add end-of-word marker
        tokens = [token_lookup[BYTE_TO_CHAR_MAP[b]] for b in word.encode('utf-8')]
        tokens.append(token_lookup[eow_marker])
        
        chain = TokenChain(tokens, word_id)
        word_chains[word] = (chain, word_id)
        
        # Initial pair counting for this unique word
        for node_one, node_two in chain.get_current_pairs():
            manager.add_occurrence(node_one)

    # 4. Adjust pair counts based on word frequency
    for word, freq in word_counts.items():
        if freq > 1:
            chain, _ = word_chains[word]
            for node_one, node_two in chain.get_current_pairs():
                pair = (node_one.token_id, node_two.token_id)
                manager.pair_frequencies[pair] += freq - 1

    # 5. Build priority queue
    manager.create_heap()
    
    # 6. Main merge loop
    merge_rules = []
    num_merges_needed = vocab_size - len(token_lookup)
    
    word_id_to_freq_map = {i: freq for i, (word, freq) in enumerate(word_counts.items())}
    
    for i in range(num_merges_needed):
        top_pair = manager.get_top_pair()
        if top_pair is None:
            print(f"Stopping early: No more pairs to merge after {i} merges.")
            break
        
        # Add new token to vocabulary
        new_token_id = len(token_lookup)
        tok1, tok2 = top_pair
        new_token_string = id_lookup[tok1] + id_lookup[tok2]
        
        token_lookup[new_token_string] = new_token_id
        id_lookup[new_token_id] = new_token_string
        merge_rules.append(top_pair)
        
        # Apply the merge across all occurrences
        locations_to_merge = manager.get_valid_locations(top_pair)
        pairs_to_update = set()
        
        for start_node in locations_to_merge:
            end_node = start_node.next_node
            chain = start_node.parent_chain
            word_freq = word_id_to_freq_map.get(chain.word_ref_id, 1)

            # Decrement counts of pairs that will be broken by this merge
            if start_node.prev_node and start_node.prev_node.is_active:
                old_pair = (start_node.prev_node.token_id, start_node.token_id)
                manager.pair_frequencies[old_pair] -= word_freq
                pairs_to_update.add(old_pair)
            
            if end_node.next_node and end_node.next_node.is_active:
                old_pair = (end_node.token_id, end_node.next_node.token_id)
                manager.pair_frequencies[old_pair] -= word_freq
                pairs_to_update.add(old_pair)
            
            # Perform the merge
            new_node = chain.perform_merge(start_node, end_node, new_token_id)
            
            # Increment counts for newly formed pairs
            if new_node:
                if new_node.prev_node and new_node.prev_node.is_active:
                    new_pair = (new_node.prev_node.token_id, new_node.token_id)
                    manager.pair_frequencies[new_pair] += word_freq
                    pairs_to_update.add(new_pair)
                
                if new_node.next_node and new_node.next_node.is_active:
                    new_pair = (new_node.token_id, new_node.next_node.token_id)
                    manager.pair_frequencies[new_pair] += word_freq
                    pairs_to_update.add(new_pair)
        
        # Update the priority queue with all affected pairs
        for pair in pairs_to_update:
            manager.update_heap(pair)

    # 7. Finalize vocabulary list
    final_vocab = []
    final_vocab.extend(special_tokens)
    final_vocab.append(eow_marker)
    
    # Add byte tokens not already included
    byte_tokens = [BYTE_TO_CHAR_MAP[i] for i in range(256)]
    for tok in byte_tokens:
        if tok not in final_vocab:
            final_vocab.append(tok)
            
    # Add merged tokens
    for tok1, tok2 in merge_rules:
        merged_str = id_lookup[tok1] + id_lookup[tok2]
        final_vocab.append(merged_str)
        
    tokenizer_model = BPEModel(merge_rules, id_lookup)
    return final_vocab, tokenizer_model

def tokenize(text, tokenizer):
    eow_marker = "</w>"
    final_tokens = []
    
    words = text.split()
    for word in words:
        # Start with byte-level representation
        current_tokens = [BYTE_TO_CHAR_MAP[b] for b in word.encode('utf-8')] + [eow_marker]
        
        # Iteratively apply learned merge rules in order
        for tok1_id, tok2_id in tokenizer.merge_rules:
            tok1_str = tokenizer.id_to_token_map[tok1_id]
            tok2_str = tokenizer.id_to_token_map[tok2_id]
            merged_str = tok1_str + tok2_str
            
            i = 0
            new_tokens = []
            while i < len(current_tokens):
                # Check for a potential merge at the current position
                if i + 1 < len(current_tokens) and current_tokens[i] == tok1_str and current_tokens[i+1] == tok2_str:
                    new_tokens.append(merged_str)
                    i += 2
                else:
                    new_tokens.append(current_tokens[i])
                    i += 1
            current_tokens = new_tokens
            
        final_tokens.extend(current_tokens)
        
    return final_tokens

def detokenize(tokens, tokenizer):
    eow_marker = "</w>"
    buffer = bytearray()
    
    for token in tokens:
        if token.endswith(eow_marker):
            # Handle end of a word
            prefix = token[:-len(eow_marker)]
            if prefix:
                for char in prefix:
                    if char in CHAR_TO_BYTE_MAP:
                        buffer.append(CHAR_TO_BYTE_MAP[char])
            buffer.extend(b' ') # Add a space after each word
        elif token not in ["<pad>", "<unk>", "<s>", "</s>"]:
            # Handle regular tokens
            for char in token:
                if char in CHAR_TO_BYTE_MAP:
                    buffer.append(CHAR_TO_BYTE_MAP[char])

    return buffer.strip().decode('utf-8', errors='replace')
    
def load_training_data(train_path):
    with open(train_path, "r", encoding="utf-8") as f:
        return f.read()

def save_vocab(vocab, rollno, vocab_size):
    fname = f"{rollno}_assignment2_bpe_vocab_{vocab_size}.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for token in vocab:
            f.write(token + "\n")
    print(f"Vocabulary of size {len(vocab)} saved to {fname}")

def save_tokens(tokens, rollno):
    fname = f"{rollno}_assignment2_bpe_tokens.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for tok in tokens:
            f.write(tok + "\n")
    print(f"Tokens saved to {fname}")

def save_detokenized(text, rollno):
    fname = f"{rollno}_assignment2_bpe_detokenized.txt"
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
    print(f"Training BPE tokenizer for vocab size {args.vocab_size}...")
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