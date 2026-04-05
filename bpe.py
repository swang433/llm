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
        self.bpe_ranks = {} #FIX ME !!!
    
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
            #increasing index by 2 if replaced, 1 if not replaced
            if id_index < len(ids) - 1 and (ids[id_index], ids[id_index + 1]) == pair_id: 
                replaced.append(new_id)
                id_index += 2
            else: 
                replaced.append(ids[id_index])
                id_index += 1
        return replaced
    
    def encode(self, text, allowed_special=None): 
        '''
        args: text for encoding, possibly enabling special tokens
        return: list of token ids 
        '''
        import re
        token_ids = []
        
        if allowed_special is not None and len(allowed_special) > 0:
            pattern = '(' + '|'.join(re.escape(tok) for tok in sorted(allowed_special, key=len, reverse=True)) + ')'
            match_index = 0
            for match in re.finditer(pattern, text): 
                prefix = text[match_index:match.start()]
                token_ids.extend(self.encode(prefix, allowed_special=None)) #recursive
                special_token = match.group(0) #finds the first matched pattern
                if special_token in self.inv_vocab: 
                    token_ids.append(self.inv_vocab[special_token])
                else: 
                    raise(ValueError(f'Special token "{special_token} not found in vocabulary'))
                match_index = match.end()
            text = text[match_index:] #porcess the rest of the text normally
            
            disallowed = [
                tok for tok in self.inv_vocab
                if tok.startswith('<|') and tok.endswith('|>') and tok in text and tok not in allowed_special
            ]
            if disallowed: 
                raise(ValueError(f'Disallowed token found in text: {disallowed}'))

        #normal processing without special tokens
        tokens = re.split(r'([,.:;!_?"()\']|--|\s)', text)
        for token in tokens: 
            token = token.strip()
            if not token: 
                continue
            elif token in self.vocab:
                token_ids.append(token)
            else: 
                token_ids.extend(self.bpe_tokenize(token))
                
        return token_ids
    
    def bpe_tokenize(self, token): 
        """
        Tokenize a single token using BPE merges.

        Args:
            token (str): The token to tokenize.

        Returns:
            List[int]: The list of token IDs after applying BPE.
        """
        #handles OOV
        token_ids = [self.inv_vocab.get(char, None) for char in token]
        if None in token_ids: 
            missing = [c for c, id in zip(token, token_ids) if id is None]
            raise(ValueError(f'Characters not found in vocabulary: {missing}'))

        #handles ranking
        if not self.bpe_ranks: 
            merge = True
            while merge and len(token_ids) > 1: 
                merge = False
                new_tokens = []
                i = 0
                while i < len(token_ids) - 1: 
                    pair = (token_ids[i], token_ids[i + 1])
                    if pair in self.bpe_merges: 
                        merged_token_id = self.bpe_merges[pair]
                        new_tokens.append(merged_token_id)
                        i += 2
                        merge = True
                    else:
                        new_tokens.append(token_ids[i]) 
                        i += 1
                if i < len(token_ids): 
                    new_tokens.append(token_ids[i])
                token_ids = new_tokens
        return token_ids
    
    def save_vocab_and_merges(self, vocab_path, bpe_merges_path):
        '''
        Save the vocabulary and BPE merges to JSON files.

        Args:
            vocab_path (str): Path to save the vocabulary.
            bpe_merges_path (str): Path to save the BPE merges.
        '''
        # Save vocabulary
        with open(vocab_path, "w", encoding="utf-8") as file:
            json.dump(self.vocab, file, ensure_ascii=False, indent=2)

        # Save BPE merges as a list of dictionaries
        with open(bpe_merges_path, "w", encoding="utf-8") as file:
            merges_list = [{"pair": list(pair), "new_id": new_id}
                            for pair, new_id in self.bpe_merges.items()]
            json.dump(merges_list, file, ensure_ascii=False, indent=2)  

    def load_vocab_and_merges(self, vocab_path, bpe_merges_path):
        '''
        Load the vocabulary and BPE merges from JSON files.

        Args:
            vocab_path (str): Path to the vocabulary file.
            bpe_merges_path (str): Path to the BPE merges file.
        '''
        # Load vocabulary
        with open(vocab_path, "r", encoding="utf-8") as file:
            loaded_vocab = json.load(file)
            self.vocab = {int(k): v for k, v in loaded_vocab.items()}
            self.inv_vocab = {v: int(k) for k, v in loaded_vocab.items()}

        # Load BPE merges
        with open(bpe_merges_path, "r", encoding="utf-8") as file:
            merges_list = json.load(file)
            for merge in merges_list:
                pair = tuple(merge["pair"])
                new_id = merge["new_id"]
                self.bpe_merges[pair] = new_id
                
    @lru_cache(maxsize=None)
    def get_special_token_id(self, token):
        return self.inv_vocab.get(token, None)