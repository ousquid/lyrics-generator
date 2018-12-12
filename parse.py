import sys
import re
import MeCab 
import os

mcb = MeCab.Tagger("-d /usr/lib64/mecab/dic/ipadic") 

for fname in os.listdir(sys.argv[1]):
    fpath = os.path.join(sys.argv[1], fname)
    opath = os.path.join(sys.argv[2], fname)
    with open(fpath) as f:
        txt = f.read()
        txt = txt.replace('\n', ' ')
        txt = txt.replace('　', ' ')
        txt = re.sub(r"\s+", " ", txt)
 
    parse_text = mcb.parse(txt) 
    with open(opath, "w") as f:
        tokens = list()
        for i in parse_text.split("\n"): 
            if i == "EOS": break
            f.write(i.split()[0])
            f.write(" ")