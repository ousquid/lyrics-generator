import sys
from janome.tokenizer import Tokenizer
with open(sys.argv[1]) as f:
    txt = f.read()
for token in Tokenizer().tokenize(txt):
    print(token.surface, end=' ')