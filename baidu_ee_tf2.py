# -*- coding: utf-8 -*-
"""

-------------------------------------------------
   File Name：        baidu_ee_hf
   Description :
   Author :           heyouxin
   Create date：      2021/5/12
   Latest version:    v1.0.0
-------------------------------------------------
   Change Log:
#-----------------------------------------------#
    v1.0.0            hyx        2021/5/12
    1.
#-----------------------------------------------#
-------------------------------------------------

"""

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import json
import tensorflow_text as text
from tqdm import tqdm
#from my_function import DataGenerator

tf.get_logger().setLevel('ERROR')
def load_data(filename):
    D = []
    with open(filename) as f:
        for l in f:
            l = json.loads(l)
            D.append(l)
    return D

def search(pattern,sequence,d):
    n = len(pattern)
    for i in range(len(sequence)):
        try:
            if (sequence[i:i+n] == pattern).all():
                return i
        except:
            return -1
    return -1

'''
class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        x, y = [], []

        for is_end, d in self.sample(random):
            text = d['text']
            text_tok = tokenizer([text]).numpy()[0]
            y_label = [0 for _ in range(len(text_tok))]
            for e in d['event_list']:
                trigger_tok = tokenizer([e['trigger']]).numpy()[0]
                t_idx = search(trigger_tok,text_tok,d)
                if t_idx != -1:
                    y_label[t_idx:t_idx+len(trigger_tok)] = [label2id[e['event_type']]] * len(trigger_tok)
                for a in e['arguments']:
                    a_tok = tokenizer([a['argument']]).numpy()[0]
                    a_idx = search(a_tok,text_tok,d)
                    if a_idx != -1:
                        y_label[a_idx:a_idx+len(a_tok)] = [label2id[e['event_type']+ '_'+a['role']]] * len(a_tok)
            x.append(text)
            y.append([0]+y_label[0:126])

          
            if len(x) == self.batch_size or is_end:
                yield x,y
                x,y = [], []
'''

def general_y_label(train_data,tokenizer):
    y = []
    x = []
    for d in tqdm(train_data):
        text = d['text']
        text_tok = tokenizer([text]).numpy()[0]
        y_label = [0 for _ in range(len(text_tok))]
        for e in d['event_list']:
            trigger_tok = tokenizer([e['trigger']]).numpy()[0]
            t_idx = search(trigger_tok,text_tok,d)
            if t_idx != -1:
                y_label[t_idx:t_idx+len(trigger_tok)] = [label2id[e['event_type']]] * len(trigger_tok)
            for a in e['arguments']:
                a_tok = tokenizer([a['argument']]).numpy()[0]
                a_idx = search(a_tok,text_tok,d)
                if a_idx != -1:
                    y_label[a_idx:a_idx+len(a_tok)] = [label2id[e['event_type']+ '_'+a['role']]] * len(a_tok)
        x.append(text)

        y.append([0]+y_label)
    return x,y

def build_seq_tag_model():
    text_input = tf.keras.layers.Input(shape=(),dtype = tf.string,name='text')
    preprocessing_layer = hub.KerasLayer(tfhub_preprocess,name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(tfhub_encoder,trainable=True,name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['sequence_output']
    net = tf.keras.layers.Dense(len(label2id),activation='softmax',name='seq-tag')(net)
    return tf.keras.Model(text_input,net)


if __name__ == '__main__':

    tfhub_preprocess = 'bert_txt_preprocess_cn/'
    bert_preprocess_model = hub.KerasLayer(tfhub_preprocess)

    tfhub_encoder = './bert_cn/'
    bert_model = hub.KerasLayer(tfhub_encoder)

    bert_precess = hub.load(tfhub_preprocess)
    tokenizer = hub.KerasLayer(bert_precess.tokenize,name = 'tokenizer')


    # 读取数据
    train_data = load_data('./datasets/DuEE/train.json')
    valid_data = load_data('./datasets/DuEE/dev.json')

    # 读取schema
    with open('./datasets/DuEE/event_schema.json') as f:
        id2label = {0:0}
        label2id = {0:0}
        n = 1
        for l in f:
            l = json.loads(l)
            id2label[n] = l['event_type']
            label2id[l['event_type']] = n
            n += 1
            for role in l['role_list']:
                key = l['event_type']+'_'+role['role']
                id2label[n] = key
                label2id[key] = n
                n += 1

    '''
    train_x_4,train_y_4 = general_y_label(train_data[9000:len(train_data)],tokenizer)
    tmp_d = {'x':train_x_4,'y':train_y_4}
    import pandas as pd
    tmp_df = pd.DataFrame(tmp_d)
    tmp_df.to_csv('tmp_df_4.csv',index=False)
    '''

    import pandas as pd
    tmp_df_1 = pd.read_csv('tmp_df_1.csv')
    tmp_df_2 = pd.read_csv('tmp_df_2.csv')
    tmp_df_3 = pd.read_csv('tmp_df_3.csv')
    tmp_df_4 = pd.read_csv('tmp_df_4.csv')
    tmp_df = pd.concat([tmp_df_1,tmp_df_2,tmp_df_3,tmp_df_4])

    y = tmp_df['y']
    train_x = list(tmp_df['x'])
    train_y = []
    for k in y:
        train_y.append(eval(k))
    train_y = tf.keras.preprocessing.sequence.pad_sequences(train_y, maxlen=128, padding='post',
                                                            truncating='post')
    for i in range(len(train_y)):
        train_y[i][127] = 0
    train_y = tf.keras.utils.to_categorical(train_y, num_classes=len(label2id))
    train_ds = tf.data.Dataset.from_tensor_slices(((train_x,train_y))).shuffle(100).batch(16)
    #batch_size = 16
    #train_generator = data_generator(train_data, batch_size)

    seq_tag_model = build_seq_tag_model()
    seq_tag_model.summary()



    '''
    from official.nlp import optimization
    epochs = 5
    steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1 * num_train_steps)

    init_lr = 3e-5
    optimizer = optimization.create_optimizer(init_lr=init_lr,
                                              num_train_steps=num_train_steps,
                                              num_warmup_steps=num_warmup_steps,
                                              optimizer_type='adamw')
    '''


    history = seq_tag_model.fit(train_ds,epochs = epochs)



    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    metrics = tf.metrics.CategoricalAccuracy()

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    seq_tag_model.compile(optimizer=optimizer,loss=loss,metrics=metrics)
    history = seq_tag_model.fit(train_ds,epochs = 5)

    seq_tag_model.predict([train_x[0]]).argmax(-1)
    seq_tag_model.save('./mymodel/')



    #---------------inference---------------------------
    seq_tag_model = tf.keras.models.load_model('./mymodel/')
    bert_preprocess_model([valid_data[500]['text']])
    y_pred = seq_tag_model.predict([valid_data[500]['text']])
    y_label = np.argmax(y_pred, axis=-1)
    y_label = y_label.reshape(1, -1)[0]
    y_ner = [id2label[i] for i in y_label]
    print(y_ner)

    bert_tokenizer_params = dict(lower_case=True)
    tokenizer = text.BertTokenizer('bert_txt_preprocess_cn/assets/vocab.txt')
    (tokens, start_offsets, end_offsets) = tokenizer.tokenize_with_offsets(["那里有4000人"])
    tokenizer.lookup(tokens)
    tokenizer.split_with_offsets(["那里有4000人"])

    token_mapping([{}])