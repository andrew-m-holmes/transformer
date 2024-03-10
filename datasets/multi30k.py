import spacy
from string import punctuation
from torchtext.data import Field
from torchtext.datasets import Multi30k
from utils.functional import write

defaults = {"source": spacy.load("en_core_web_sm"),
            "target": spacy.load("de_core_news_sm"),
            "exts": (".en", ".de"),
            "sos": "<sos>",
            "eos": "<eos>",
            "pad": "<pad>",
            "unk": "<unk>"}

class Tokenizer:

    def __init__(self, exts=None, sos=None, eos=None, pad=None, unk=None, encoder=None, decoder=None) -> None:
        self.init(exts, sos, eos, pad, unk, encoder, decoder)
    
    def tokenize_source(self, text):
        tokens = [tok.text for tok in self.encoder.tokenizer(text)]
        return tokens

    def tokenize_target(self, text):
        tokens = [tok.text for tok in self.decoder.tokenizer(text)]
        return tokens
    
    def init(self, exts, sos, eos, pad, unk, encoder, decoder):
        # defaults
        if exts is None:
            exts = defaults["exts"]
        if sos is None:
            sos = defaults["sos"]
        if eos is None:
            eos = defaults["eos"]
        if pad is None:
            pad = defaults["pad"]
        if unk is None:
            unk = defaults["unk"]
        if encoder is None:
            encoder = defaults["source"]
        if decoder is None:
            decoder = defaults["target"]
        # create fields
        self.exts, self.sos, self.eos, self._pad, self.unk, self.encoder, self.decoder = \
            exts, sos, eos, pad, unk, encoder, decoder
        self.special_tokens = set([sos, eos, pad, unk])
        self.source = Field(init_token=sos, eos_token=eos, pad_token=pad, unk_token=unk,  
                            lower=True, tokenize=self.tokenize_source, batch_first=True)
        self.target = Field(init_token=sos, eos_token=eos, pad_token=pad, unk_token=unk, 
                            lower=True, tokenize=self.tokenize_target, batch_first=True)
        self.fields = self.source, self.target

def convert_tokens_to_string(tokens):
    text = " ".join(tokens)
    for punct in punctuation:
        text = text.replace(" " + punct, punct)
    return text
    
if __name__ == "__main__":
    # create tokenizer & retrieve data
    tokenizer = Tokenizer()
    traindata, testdata, valdata = Multi30k.splits(exts=tokenizer.exts, fields=tokenizer.fields)

    # go through english & german samples
    for data in testdata:
        src, trg = data.src, data.trg
        # parse & process sample pairs
        en, de = convert_tokens_to_string(src), convert_tokens_to_string(trg)
        # write postprocess to dataset .txt file
        write(en, path="datasets/multi30k-test.en")
        write(de, path="datasets/multi30k-test.de")