import sys
import re
import os

for fname in os.listdir(sys.argv[1]):
    fpath = os.path.join(sys.argv[1], fname)
    opath = os.path.join(sys.argv[2], fname)
    with open(fpath) as f:
        txt = f.read()
        txt = re.sub("\(.*?\)|（.*?）", "", txt)
        txt = re.sub("[^ \u30fc\u3041-\u3093\u30a1-\u30f6\u4e00-\u9fa0]", "", txt)
        txt = re.sub(r"\s+", " ", txt)

    with open(opath, "w") as f:
        f.write(txt)
