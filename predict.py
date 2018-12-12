import sys
import os
import MeCab
import numpy

from keras.models import Model
from keras.models import load_model
from keras.layers import Input

from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors

from seq2seq import SimpleSeq2Seq, Seq2Seq, AttentionSeq2Seq
import numpy as np
import pickle

input_length = 5
output_length = 14

def parse(text):
    mcb = MeCab.Tagger("-d /usr/lib64/mecab/dic/ipadic")
    parsed_text = mcb.parse(text) 
    tokens = list()
    for i in parsed_text.split("\n"): 
        if i == "EOS": break
        tokens.append(i.split()[0])
    
    return tokens[0:input_length]


def get_distribute_vector(distributes, input_text):
    word_list = parse(input_text)
    print(word_list)
    return numpy.array([ distributes.get_vector(word) \
                        if word in distributes.vocab.keys() \
                        else distributes.get_vector("トークン") \
                        for word in word_list])


if __name__=="__main__":
    if len(sys.argv)<3:
        print("usage: {} input_model input_text".format(sys.argv[0]))
        exit()
    input_model = sys.argv[1]
    input_text = sys.argv[2]
    
    dist_path='./jawiki.all_vectors.100d.txt'
    if not os.path.exists("tmp.pkl"):
        distributes = KeyedVectors.load_word2vec_format(dist_path)
        with open("tmp.pkl", "wb") as f:
            pickle.dump(distributes, f)
    else:
        with open("tmp.pkl", "rb") as f:
            distributes = pickle.load(f)

    dist_vec = get_distribute_vector(distributes, input_text)
    
    hidden_dim=512
    depth_num=3
    input_dim = 100
    output_dim = 100
    model = AttentionSeq2Seq(
        input_shape=(input_length, input_dim), depth=depth_num,
        output_dim=output_dim, hidden_dim=hidden_dim, output_length=output_length)

    model.compile(loss='mse', optimizer="rmsprop")
    model.load_weights(input_model)
    y_pred = model.predict(numpy.array([dist_vec]))
    for i in y_pred[0]:
        print(distributes.similar_by_vector(i, topn=1)[0][0], end=" ")