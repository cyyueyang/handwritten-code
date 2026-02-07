import re
import json
import os

def read_merges(merges_path):
    with open(merges_path, 'r', encoding="utf-8") as f:
        lines = f.readlines()

    merges = []

    for line in lines:
        pair = line.strip().split()

        if len(pair) == 2:
            merges.append(pair)

    return merges

def read_vocab(vocab_path):
    with open(vocab_path, 'r', encoding="utf-8") as f:
        vocab = json.load(f)

    return vocab

class Tokenizer:
    def __init__(self, path):
        merges_path = os.path.join(path, 'merges.txt')
        vocab_path = os.path.join(path, 'vocab.json')
        self.merges = read_merges(merges_path)
        self.vocab = read_vocab(vocab_path)

        self.space_token = 'Ġ'
        self.newline_token = 'Ċ'
        self.vocab['<|im_start|>'] = 151644
        self.vocab['<|im_end|>'] = 151645
        self.id2token = {v: k for k, v in self.vocab.items()}

    def tokenize(self, text):
        tokens = []
        pattern = r'<\|\w+\|>'
        matches = re.finditer(pattern, text)

        t = 0
        for match in matches:
            match_text = match.group()
            pre_text = text[t: match.start()]
            if pre_text:
                tokens.extend(self._tokenize(pre_text))
            tokens.append(match_text)
            t = match.end()
        if t < len(text):
            tokens.extend(self._tokenize(text[t:]))
        return tokens

    def _tokenize(self, text):
        text = text.replace(' ', self.space_token)
        text = text.replace('\n', self.newline_token)

        tokens = list(text)

        for pair in self.merges:
            first, second = pair
            new_token = first + second

            new_tokens = []
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == first and tokens[i + 1] == second:
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            if i < len(tokens):
                new_tokens.append(tokens[i])

            tokens = new_tokens

        return tokens

    def encode(self, text):
        tokens = self.tokenize(text)
        return [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]

    def decode(self, tokens_ids):
        tokens = [self.id2token[id] for id in tokens_ids]
        text = "".join(tokens)
        text = text.replace(self.space_token, ' ')
        text = text.replace(self.newline_token, '\n')
        return text

if __name__ == '__main__':
    from transformers import AutoTokenizer
    hf_tokenizer = AutoTokenizer.from_pretrained(r"./Qwen/Qwen2.5-0.5B-Instruct")

    prompt = "give me a short introduction to large language model"

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    text = hf_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    print(text)

    tokens_hf = hf_tokenizer.tokenize(text)
    input_ids_hf = hf_tokenizer([text], return_tensors="pt")['input_ids'][0].tolist()

    tokenizer_self = Tokenizer(r"./Qwen/Qwen2.5-0.5B-Instruct")
    tokens_self = tokenizer_self.tokenize(text)
    input_ids_self = tokenizer_self.encode(text)

    if tokens_self == tokens_hf :
        print("tokens test passed")

    if input_ids_self == input_ids_hf :
        print("input_ids test passed")





