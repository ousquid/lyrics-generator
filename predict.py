import sys
import os
import MeCab
import numpy
import argparse

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
    parser = argparse.ArgumentParser(description='Seq2Seq Predict.')
    parser.add_argument('input_model', type=str)
    parser.add_argument('input_text', type=str)
    parser.add_argument('-i', '--iter', type=int, default=10)
    parser.add_argument('--dist_path', type=str, default='./jawiki.all_vectors.100d.txt')
    args = parser.parse_args()
        
    if not os.path.exists("tmp.pkl"):
        distributes = KeyedVectors.load_word2vec_format(args.dist_path)
        with open("tmp.pkl", "wb") as f:
            pickle.dump(distributes, f)
    else:
        with open("tmp.pkl", "rb") as f:
            distributes = pickle.load(f)

    middle_dim, depth_num, _, input_length, output_length, _ = [int(i) for i in args.input_model.split(".")[:-1]]
    dist_vec = get_distribute_vector(distributes, args.input_text)[:input_length, :]
    
    input_dim = 100
    output_dim = 100
    model = AttentionSeq2Seq(
        input_shape=(input_length, input_dim), depth=depth_num,
        output_dim=output_dim, hidden_dim=middle_dim, output_length=output_length)

    model.compile(loss='mse', optimizer="rmsprop")
    model.load_weights(args.input_model)
    y_pred = numpy.array([dist_vec])
    for i in range(args.iter):
        y_pred = model.predict(y_pred[:,-input_length:,:])
        for j in y_pred[0]:
            print(distributes.similar_by_vector(j, topn=1)[0][0], end=" ")