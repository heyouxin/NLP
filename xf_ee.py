# -*- coding: utf-8 -*-
"""

-------------------------------------------------
   File Name：        xf_ee
   Description :
   Author :           heyouxin
   Create date：      2020/7/25
   Latest version:    v1.0.0
-------------------------------------------------
   Change Log:
#-----------------------------------------------#
    v1.0.0            hyx        2020/7/25
    1.
#-----------------------------------------------#
-------------------------------------------------

"""
import pandas as pd
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
from CRF import CRF
from tensorflow.keras.layers import Attention,Dense
from bert import BertModelLayer
import bert
import re
from tensorflow.keras.models import Model
print(tf.__version__)
print(tf.keras.__version__)


def get_x_y(df):
    x = list(df['news'])[0]
    trigger = list(set(df['trigger']))
    y = [0] * len(x)
    for t in trigger:
        m = re.search(t, x)
        y[(m.start(0)):(m.end(0))] = [1] * len(t)
    return y

if __name__ == '__main__':


    config_path  = 'bert_chinese/albert_base_zh/albert_base/albert_config.json'
    checkpoint_path = 'bert_chinese/albert_base_zh/albert_base/model.ckpt'
    char_vocab_path = 'bert_chinese/albert_base_zh/albert_base/vocab_chinese.txt'

    with open(char_vocab_path,'r',encoding='utf8') as f:
        char_vocabs = [line.strip() for line in f]

    id2vocab = {idx:char for idx,char in enumerate(char_vocabs)}
    vocab2idx = {char:idx for idx,char in id2vocab.items()}

    ##
    id2label,label2id = {},{}
    id2label[0] = 0
    id2label[1] = 'tri'

    label2id = dict(zip(id2label.values(),id2label.keys()))
    MAX_LEN = 130
    VOCAB_SIZE = len(vocab2idx)
    CLASS_NUMS = len(label2id)

    ##
    import os
    os.path
    train = pd.read_csv('./data/xunfei/train/train.csv')
    train_x = train.iloc[:6000,:]
    val_x = train.iloc[6000:len(train),:]

    train_y = train_x.groupby('id').apply(get_x_y)
    train_x = list(train_x.drop_duplicates(subset=['id'])['news'])

    dev_y = val_x.groupby('id').apply(get_x_y)
    dev_x = list(val_x.drop_duplicates(subset=['id'])['news'])

    def get_pad_data(train_x,train_y,vocab2idx,MAX_LEN=100):
        train_datas = []
        for d in train_x:
            x = [vocab2idx.get(c,0) for c in d]
            train_datas.append(x)

        train_labels = train_y
        train_datas = keras.preprocessing.sequence.pad_sequences(train_datas,maxlen=MAX_LEN,padding='post',truncating='post')
        train_labels = keras.preprocessing.sequence.pad_sequences(train_labels,maxlen=MAX_LEN,padding='post',truncating='post')
        print(train_labels)
        print('x_train shape:',train_datas.shape)
        train_labels = keras.utils.to_categorical(train_labels,CLASS_NUMS)
        print('train label shape:', train_labels.shape)

        return train_datas,train_labels


    train_datas, train_labels = get_pad_data([[2,2],[1,1]], [[0,1,2],[1,2,0],[2,1,0]], vocab2idx, MAX_LEN)
    train_datas,train_labels = get_pad_data(train_x,train_y,vocab2idx,MAX_LEN)
    dev_datas, dev_labels = get_pad_data(dev_x, dev_y, vocab2idx, MAX_LEN)
    con_train = tf.data.Dataset.from_tensor_slices((train_datas,train_labels))
    con_dev = tf.data.Dataset.from_tensor_slices((dev_datas, dev_labels))

    train_batches = con_train.shuffle(100).batch(32)
    val_batches = con_dev.shuffle(100).batch(32)


    from bert.loader import StockBertConfig,load_stock_weights

    model_dir = 'bert_chinese/albert_base_zh/albert_base'
    bert_params = bert.params_from_pretrained_ckpt(model_dir)
    l_bert = bert.BertModelLayer.from_params(bert_params,name='albert')

    l_input_ids = keras.layers.Input(shape=(MAX_LEN,MAX_LEN),dtype='int32')
    output = l_bert(l_input_ids)

    output = Dense(CLASS_NUMS)(output)

    model_step1 = tf.keras.Model(inputs=l_input_ids, outputs=output)
    load_stock_weights(l_bert,checkpoint_path)
    model_step1.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),optimizer=tf.keras.optimizers.Adam(1e-4),metrics=['accuracy'])
    model_step1.summary()

    checkpoint_filepath = './callbacks/checkpoints/xf_ee_step1_bert_checkpoint'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,save_weight_only=True,monitor='val_accuracy',mode='max',
                                                                   verbose=1,save_best_only=True)


    #model_step1.load_weights(checkpoint_filepath)
    history = model_step1.fit(train_batches,epochs=10,validation_data=val_batches,callbacks=[model_checkpoint_callback,])


    tri_all = []
    for i in range(0,len(dev_x)):
        test__ = np.array([dev_datas[i]])
        y_pred = model_step1.predict(test__)
        y_label= np.argmax(y_pred,axis=-1)
        y_label = y_label.reshape(1,-1)[0]
        y_ner = [id2label[i] for i in y_label]

        l_all = []

        idx_s = 0
        for m in range(len(y_label)):
            if (y_label[m] == 1 ) & (idx_s == 0):
                idx_s = m
            if (idx_s != 0) & (y_label[m] != 0):
                idx_e = m
                l_all.append([idx_s,idx_e])
                idx_s = 0

        tri = []
        for j in range(len(l_all)):
            tri.append(dev_x[i][l_all[j][0]:l_all[j][1]])
        tri_all.append(tri)

    d = {'news':dev_x,'trigger':tri_all}
    df_pred = pd.DataFrame(d)


    def com_tri(df):
        df = df.reset_index(drop=True)
        news = list(df['news'])[0]
        trigger = list(df['trigger'])
        d = {'news':news,'trigger':[trigger]}
        return pd.DataFrame(d)

    df_raw = val_x[['id','news','trigger']].groupby('id').apply(com_tri)
    df_raw = df_raw.reset_index(drop=True)

    c = 'trigger'
    df_pred[c+'_res'] = np.where(df_raw[c] == df_pred[c],1,0)
    tri_acc = len(df_pred[(df_pred['trigger_res'] == 1)])/len(df_pred)
    print(tri_acc)












