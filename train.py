import sys
import os

from keras.models import Model
from keras.layers import Input

from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors

from seq2seq import SimpleSeq2Seq, Seq2Seq, AttentionSeq2Seq
import numpy as np
import pickle

def lyrics_dataset(distributes, input_dir, input_length=5, output_length=5, iteration=50):

    x_vecs_list = list() 
    y_vecs_list = list()
    for name in os.listdir(input_dir):
        input_path = os.path.join(input_dir, name)
        with open(input_path) as f:
            # dist_vecs:(items, 100)
            dist_vecs = \
                np.array([distributes.get_vector(word) 
                    if word in distributes.vocab.keys() else distributes.get_vector("?") \
                        for word in f.read().strip().split()])
            
            iter_num = max(min(len(dist_vecs)-input_length-output_length, iteration), 0)
            for i in range(iter_num):
                # x_vecs:(input_length, 100)
                x_vecs = dist_vecs[i:i+input_length, :]
                # y_vecs:(output_length, 100)
                y_vecs = dist_vecs[i+input_length:i+input_length+output_length, :]
        
                x_vecs_list.append(x_vecs)
                y_vecs_list.append(y_vecs)

    # x_vecs_list:(samples, input_length, 100)
    x_arr = np.array(x_vecs_list)
    # y_vecs_list:(samples, output_length, 100)
    y_arr = np.array(y_vecs_list)
    
    return train_test_split(x_arr, y_arr, test_size=0.0, random_state=0)


def train(x_train, y_train, epoch_num, hidden_dim=24, depth_num=1):
    input_length = x_train.shape[1]
    output_length = y_train.shape[1]
    
    input_dim = x_train.shape[2]
    output_dim = y_train.shape[2]
    model = AttentionSeq2Seq(
        input_shape=(input_length, input_dim), depth=depth_num,
        output_dim=output_dim, hidden_dim=hidden_dim, output_length=output_length)
    
    model.compile(loss='mse', optimizer="rmsprop")
    model.summary()
    
    model.fit(x_train, y_train, epochs=epoch_num)
    return model

if __name__=="__main__":
    if len(sys.argv)<6:
        print("usage: {} input_dir output_model epoch_num  input_length output_length iteration".format(sys.argv[0]))
        exit()
    input_dir = sys.argv[1]
    model_path = sys.argv[2]
    epoch_num = int(sys.argv[3])
    input_length = int(sys.argv[4])
    output_length = int(sys.argv[5])
    iteration = int(sys.argv[6])
    
    dist_path='./jawiki.all_vectors.100d.txt'
    if not os.path.exists("tmp.pkl"):
        distributes = KeyedVectors.load_word2vec_format(dist_path)
        with open("tmp.pkl", "wb") as f:
            pickle.dump(distributes, f)
    else:
        with open("tmp.pkl", "rb") as f:
            distributes = pickle.load(f)

    x_train, x_test, y_train, y_test = \
        lyrics_dataset(distributes, input_dir, input_length=input_length, output_length=output_length, iteration=iteration)
    print(x_train.shape)
    print(y_train.shape)
    model = train(x_train, y_train, epoch_num)
    model.save(model_path)
    model.evaluate(x_test, y_test)
    
    y_pred = model.predict(x_train)
    for i in y_pred[0]:
        print(distributes.similar_by_vector(i, topn=1)[0][0], end=" ")