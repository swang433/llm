from collections import Counter, deque
from functools import lru_cache
import json
import re

space = "Ġ"
class BPETokenizer(object): 
    def __init__(self):
        self.vocab = {} #char->int
        self.inv_vocab = {} #int->char
        self.bpe_merges = {}
        self.ranks = {}
    
    def train(self, text, vocab_size, allowed_special={"<|endoftext|>"}): 
        
        # Preprocess: Replace spaces with "Ġ"
        preprocessed = []
        for i, char in enumerate(text): 
            if char == ' ' and i > 0: 
                preprocessed.append(space)
            else: 
                preprocessed.append(char)
        preprocessed = ''.join(preprocessed)
        
        '''
        Initialize vocab with unique characters, including "Ġ" if present
        Start with the first 256 ASCII characters, extend with unique chars in processed text, 
        and create vocab mapping
        '''
        unique_chars = [chr(i) for i in range(256)]
        unique_chars.extend(
            [item for item in sorted(set(preprocessed)) if item not in unique_chars]
        )
        if space not in unique_chars: 
            unique_chars.append(space)
        self.vocab = {char: i for i, char in enumerate(unique_chars)} #encoding to int
        self.inv_vocab = {i: char for i, char in self.vocab.items()} #decoding to char
        
        #add special tokens
        if allowed_special:
            for tk in allowed_special: 
                if tk not in self.vocab:
                    new_id = len(unique_chars) 
                    self.vocab[tk] = new_id
                    self.inv_vocab[new_id] = tk
                    
        token_ids = [self.vocab[char] for char in preprocessed]
        
        #find pairs, repace id pairs in the list of toke ids, and populate merging dict
        for new_id in range(len(self.vocab), vocab_size):   
            pair_id = self.find_frequent_pairs(token_ids, mode='most')
            if pair_id is None: #might terminate before desired vocab size is reached
                break
            token_ids = self.replace_pair(token_ids, pair_id, new_id)
            self.bpe_merges[pair_id] = new_id
            
        
        for (id1, id2), new_id in self.bpe_merges.items(): 
            merged_tk = self.inv_vocab[id1] + self.inv_vocab[id2]
            self.inv_vocab[new_id] = merged_tk
            self.vocab[merged_tk] = new_id
            
            
    def decode(self, token_ids): 
        '''
        special tokens:
        OOV: raise value error
        new line: add space to decoded string and adds new line
        space: ignores the space character and adds the rest of the token
        '''
        decoded = ''
        for token_id in token_ids:
            if token_id not in self.inv_vocab: 
                raise ValueError('token_id not found in vocab')
            token = self.inv_vocab[token_id]
            if token == '\n': 
                if decoded and not decoded.endswith(' '): #add a space and start a new line
                    decoded += ' '
                decoded += token
            elif token.startswith(space): #discard the special space char and concatenate the rest of the token
                decoded += ' ' + token[1:]
            else: 
                decoded += token
        return decoded
    
    @staticmethod
    def find_frequent_pairs(ids, mode='most'): 
        pairs = Counter(zip(ids, ids[1:]))
        if not pairs: return None
        if mode == 'most': 
            return max(pairs.items(), key=lambda x: x[1])[0]
        if mode == 'least': 
            return min(pairs.items(), key=lambda x: x[1])[0]
        else: 
            raise ValueError('invalid mode (most or least)')
        
    @staticmethod    
    def replace_pair(ids, pair_id, new_id): #note: pair_id is a tuple of ids that were paired
        replaced, id_index = [], 0
        while id_index < len(ids): 
            if id_index < len(ids) - 1 and (ids[id_index], ids[id_index + 1]) == pair_id: 
                replaced.append(new_id)
                id_index += 2
            else: 
                replaced.append(ids[id_index])
                id_index += 1
        return replaced