import re
import yaml

with open('config.yaml') as file: 
    config = yaml.safe_load(file)

class SimpleTokenizer(object): 
    def __init__(self):
        with open(config['text']) as file: 
            raw = file.read()
        split = re.split(f'([,.:;!_?"()\']|--|\s)', raw)
        self.tokens = [item.strip() for item in split if item.strip()]
        self.words = sorted(set(self.tokens))
        self.vocab = {token: x for x, token in enumerate(self.words)}
        self.inv_vocab = {}
        for tk, num in self.vocab.items(): 
            self.inv_vocab[num] = tk
        self.vocab['<|unk|>'] = len(self.vocab)
        self.inv_vocab[self.vocab['<|unk|>']] = '<|unk|>'
            
    def encode(self, TEXT_PATH): #FIX ME: OOV token handling
        with open(TEXT_PATH) as file: 
            text = file.read()
        preprocessed = re.split(r'([,.:;!_?"()\']|--|\s)', text)
        encoding = []
        for token in preprocessed: 
            token = token.strip()
            if token in self.vocab: 
                encoding.append(self.vocab[token])
            else: 
                encoding.append(len(encoding))
        return encoding

    def decode(self, encoding): 
        return [self.inv_vocab[num] for num in encoding]