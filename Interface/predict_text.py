import codecs
import tensorflow as tf
import keras
import os
from keras_radam import RAdam
from keras import backend as K
from keras_bert import load_trained_model_from_checkpoint
import numpy as np
from keras.layers import Dense, Input, Flatten,SpatialDropout1D
from keras.layers import Conv1D, MaxPooling1D, Embedding, Concatenate, Dropout,GlobalMaxPool1D,Lambda
from keras.models import Model
from keras.layers import Bidirectional,LSTM,GRU
from keras_bert import Tokenizer
# from vncorenlp import VnCoreNLP
from pyvi.ViTokenizer import tokenize
import pickle 
import re
from gensim.utils import simple_preprocess
import pandas as pd
from nltk import flatten
import sys
tf.compat.v1.global_variables

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


class NonMasking(keras.layers.Layer): 
    def __init__(self, **kwargs):   
        self.supports_masking = True  
        super(NonMasking, self).__init__(**kwargs)   
  
    def build(self, input_shape):   
        input_shape = input_shape   
  
    def compute_mask(self, input, input_mask=None):   
        # do not pass the mask to the next layers   
        return None   
  
    def call(self, x, mask=None):   
        return x   
  
    def get_output_shape_for(self, input_shape):   
        return input_shape  
def bert_rcnn(model): 
    inputs = model.inputs
    bert_out = NonMasking()(model.outputs)

    bert_out = SpatialDropout1D(0.2)(bert_out)

    l_embedding = Lambda(lambda x: K.concatenate([K.zeros(shape=(K.shape(x)[0], 1, K.shape(x)[-1])),
                                                            x[:, :-1]], axis=1))(bert_out)
            
    r_embedding = Lambda(lambda x: K.concatenate([K.zeros(shape=(K.shape(x)[0], 1, K.shape(x)[-1])),
                                                            x[:, 1:]], axis=1))(bert_out)

    forward = LSTM(256, return_sequences=True)(l_embedding) 
    backward = LSTM(256, return_sequences=True, go_backwards=True)(r_embedding)
    backward = Lambda(lambda x: K.reverse(x, axes=1))(backward)

    together = [forward, bert_out , backward]

    together = Concatenate(axis=2)(together)

    semantic = Conv1D(256, kernel_size=1, activation="relu")(together)
    sentence_embed = Lambda(lambda x: K.max(x, axis=1))(semantic)

    dense_layer = Dense(256, activation='relu')(sentence_embed)
    preds = Dense(1, activation='sigmoid')(dense_layer)
    model_rcnn = Model(inputs, preds)
    model_rcnn.compile(loss='binary_crossentropy', optimizer=RAdam(learning_rate=2e-5), metrics=['acc'])
    model_rcnn.load_weights(resource_path('256_checkpoint_rcnn2_foody.h5'))
    return model_rcnn
    
class BERT():
    def __init__(self):
        pretrained_path = 'multi_cased_L-12_H-768_A-12'
        config_path = os.path.join(pretrained_path, 'bert_config.json')
        checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
        vocab_path = os.path.join(pretrained_path, 'vocab.txt')
        self.model = load_trained_model_from_checkpoint(config_path,checkpoint_path,seq_len=256,output_layer_num=4,trainable=True)
        self.replace_list = pickle.load(open(resource_path('replace_list.pkl'),'rb'))
        with open(resource_path('pos.txt'),'r', encoding="UTF-8") as f:   #load file từ vựng tích cực
            self.Pos_list = [i.strip() for i in f]
        f.close()
        with open(resource_path('neg.txt'), 'r', encoding="UTF-8") as f: #load file từ vựng tiêu cực
            self.Neg_list = [i.strip() for i in f]
        f.close()
        token_dict = {}    
        with codecs.open(vocab_path, 'rb','utf-8') as reader: #load file vocab vào token_dict
            for line in reader:
                token = line.strip()
                token_dict[token] = len(token_dict) 
        self.index = Tokenizer(token_dict,cased=True)
        self.bert_rcnn = bert_rcnn(self.model)

    
    
    def preprocess(self,text):   
        check = re.search(r'([a-z])\1+',text)
        if check:
            if len(check.group())>2:
                text = re.sub(r'([a-z])\1+', lambda m: m.group(1), text, flags=re.IGNORECASE) #remove các ký tự kéo dài như hayyy,ngonnnn...

        text = text.strip() #loại dấu cách đầu câu
        text = text.lower() #chuyển tất cả thành chữ thường

        text = re.sub('< a class.+</a>',' ',text)

        for k, v in self.replace_list.items():       #replace các từ có trong replace_list
            text = text.replace(k, v)       

        text = re.sub(r'_',' ',text)  
        
        text = ' '.join(i for i in flatten(tokenize(text).split(" ")))             #gán từ ghép
        
        for i in self.Pos_list:                                    #thêm feature positive
            if re.search(' '+i+' ',text): 
                text = re.sub(i,i+' positive ',text)
        for i in self.Neg_list:                                    #thêm feature negative
            if re.search(' '+i+' ',text):
                text = re.sub(i,i+' negative ',text)
        return text

    def load_data(self, text):        #đưa các từ thành index
        indices = []
        ids,_ = self.index.encode(text, max_len=256)
        indices.append(ids)

        return [indices, np.zeros_like(indices)]

    def predict_text(self, text):
        text = self.preprocess(text)     
        # print("preprocessed")      #preprocess câu đầu vào
        text_input = self.load_data(text)      #chuyển thành index       
        # print("xong index")   
        return np.round(self.bert_rcnn.predict(text_input))  #predict câu bằng BERT-RCNN

# a = BERT()
# file1 = open("test.txt","r", encoding="utf-8") 
# content = file1.read().split("\n\n")
# print(len(content))
# for i in content:
#     # print(i)
#     result = a.predict_text(i)
#     if result[0][0] == 0.0:
#         print("negative")
#     else:
#         print("positive")
# # file1.write("".join(kq))
# file1.close()