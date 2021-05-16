# -*- coding: utf-8 -*-
"""

-------------------------------------------------
   File Name：        baidu_ee_transformers
   Description :
   Author :           heyouxin
   Create date：      2021/5/14
   Latest version:    v1.0.0
-------------------------------------------------
   Change Log:
#-----------------------------------------------#
	v1.0.0            hyx        2021/5/14
	1.
#-----------------------------------------------#
-------------------------------------------------

"""

import tensorflow as tf
import json
import pylcs
from tqdm import tqdm
import numpy as np
from transformers import BertTokenizerFast, AutoTokenizer, BertTokenizer, TFBertForSequenceClassification, \
	TFBertForTokenClassification, TFAlbertModel

tf.get_logger().setLevel('ERROR')

def load_data(filename):
	D = []
	with open(filename) as f:
		for l in f:
			l = json.loads(l)
			arguments = {}
			for event in l['event_list']:
				for argument in event['arguments']:
					key = argument['argument']
					value = (event['event_type'], argument['role'])
					arguments[key] = value
			D.append((l['text'], arguments))
	return D

def predict_to_file(in_file, out_file):
	"""预测结果到文件，方便提交
	"""
	fw = open(out_file, 'w', encoding='utf-8')
	with open(in_file) as fr:
		for l in tqdm(fr):
			l = json.loads(l)
			arguments = extract_arguments(l['text'])
			event_list = []
			for k, v in arguments.items():
				event_list.append({
					'event_type': v[0],
					'arguments': [{
						'role': v[1],
						'argument': k
					}]
				})
			l['event_list'] = event_list
			l = json.dumps(l, ensure_ascii=False)
			fw.write(l + '\n')
	fw.close()


def evaluate(data):
	"""评测函数（跟官方评测结果不一定相同，但很接近）
	"""
	X, Y, Z = 1e-10, 1e-10, 1e-10
	for text, arguments in tqdm(data):
		inv_arguments = {v: k for k, v in arguments.items()}
		pred_arguments = extract_arguments(text)
		pred_inv_arguments = {v: k for k, v in pred_arguments.items()}
		Y += len(pred_inv_arguments)
		Z += len(inv_arguments)
		for k, v in pred_inv_arguments.items():
			if k in inv_arguments:
				# 用最长公共子串作为匹配程度度量
				l = pylcs.lcs(v, inv_arguments[k])
				X += 2. * l / (len(v) + len(inv_arguments[k]))
	f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
	return f1, precision, recall





class Evaluator(tf.keras.callbacks.Callback):
	"""评估和保存模型
	"""
	def __init__(self):
		self.best_val_f1 = 0.

	def on_epoch_end(self, epoch, logs=None):
		f1, precision, recall = evaluate(valid_data)
		if f1 >= self.best_val_f1:
			self.best_val_f1 = f1
			model.save_pretrained('./ee_trans/')
		print(
			'f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
			(f1, precision, recall, self.best_val_f1)
		)

def search(pattern, sequence):
	"""从sequence中寻找子串pattern
	如果找到，返回第一个下标；否则返回-1。
	"""
	n = len(pattern)
	for i in range(len(sequence)):
		if sequence[i:i + n] == pattern:
			return i
	return -1


def general_y_label(train_data,tokenizer):
	y = []
	x = []
	for d in tqdm(train_data):
		text = d[0]
		text_tok = tokenizer([text])
		token_ids = text_tok.input_ids[0]
		labels = [0 for _ in range(len(token_ids))]
		arguments = d[1]
		for argument in arguments.items():
			a_token_ids = tokenizer.encode(argument[0])[1:-1]
			start_index = search(a_token_ids, token_ids)
			if start_index != -1:
				labels[start_index] = label2id[argument[1]] * 2 + 1
				for i in range(1, len(a_token_ids)):
					labels[start_index + i] = label2id[argument[1]] * 2 + 2
		x.append(text)
		y.append(labels)
	return x,y


def extract_arguments(text):
	"""arguments抽取函数
	"""

	x_tok = tokenizer([text],is_split_into_words = True,return_offsets_mapping=True,return_tensors='tf')
	mapping = x_tok.offset_mapping[0].numpy()
	x_tok.pop('offset_mapping')

	output = model(x_tok)
	labels =np.argmax(output['logits'],axis=-1)[0]

	arguments, starting = [], False
	for i, label in enumerate(labels):
		if label > 0:
			if label % 2 == 1:
				starting = True
				arguments.append([[i], id2label[(label - 1) // 2]])
			elif starting:
				arguments[-1][0].append(i)
			else:
				starting = False
		else:
			starting = False

	return {
		text[mapping[w[0]][0]:mapping[w[-1]][-1] + 1]: l
		for w, l in arguments
	}



if __name__ == '__main__':

	# 读取数据
	train_data = load_data('./datasets/DuEE/train.json')
	valid_data = load_data('./datasets/DuEE/dev.json')

	# 读取schema
	with open('./datasets/DuEE/event_schema.json') as f:
		id2label, label2id, n = {}, {}, 0
		for l in f:
			l = json.loads(l)
			for role in l['role_list']:
				key = (l['event_type'], role['role'])
				id2label[n] = key
				label2id[key] = n
				n += 1
		num_labels = len(id2label) * 2 + 1


	model_dir = '../pretrain_model/bert-base-chinese'
	tokenizer = BertTokenizerFast.from_pretrained(model_dir,model_max_length =128,max_length = 128,padding_side = 'right',padding='max_length',truncation='True')

	x,y = general_y_label(train_data,tokenizer)
	train_x = tokenizer(x, padding=True, truncation=True, return_tensors='tf')
	train_y = tf.keras.preprocessing.sequence.pad_sequences(y,maxlen=128,padding='post',truncating='post')
	train_ds = tf.data.Dataset.from_tensor_slices(((dict(train_x), train_y))).shuffle(100).batch(8)

	#model
	model = TFBertForTokenClassification.from_pretrained(model_dir,num_labels=num_labels)
	optimizer = tf.keras.optimizers.Adam()
	loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
	metrics = tf.keras.metrics.SparseCategoricalAccuracy('sparse_accuracy')
	model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])
	model.summary()

	evaluator = Evaluator()
	history = model.fit(train_ds,epochs=1,callbacks=[evaluator])
	#model.save_pretrained('./ee_trans/')

	# ----inference----
	x_pred = tokenizer(x[0],padding='True',truncation='True',return_tensors='tf')
	output = model(x_pred)
	y_label = np.argmax(output['logits'],axis=-1)
	extract_arguments(x[0])





'''
train_x_1,train_y_1 = general_y_label(train_data[0:3000],tokenizer)
tmp_d = {'x':train_x_1,'y':train_y_1}
import pandas as pd
tmp_df = pd.DataFrame(tmp_d)
tmp_df.to_csv('tmp_df_1.csv',index=False)


#--数据
import pandas as pd

tmp_df_1 = pd.read_csv('tmp_df_1.csv')
tmp_df_2 = pd.read_csv('tmp_df_2.csv')
tmp_df_3 = pd.read_csv('tmp_df_3.csv')
tmp_df_4 = pd.read_csv('tmp_df_4.csv')
tmp_df = pd.concat([tmp_df_1, tmp_df_2, tmp_df_3, tmp_df_4])

y = tmp_df['y']
train_x = list(tmp_df['x'])
train_y = []
for k in y:
	train_y.append(eval(k))
train_y = tf.keras.preprocessing.sequence.pad_sequences(train_y, maxlen=128, padding='post',
										truncating='post')
'''

from tensorflow import keras
import tensorflow.keras as K
from tensorflow.keras.layers import Layer

class ConditionalRandomField(Layer):
    """纯Keras实现CRF层
    CRF层本质上是一个带训练参数的loss计算层。
    """
    def __init__(self, lr_multiplier=1, **kwargs):
        super(ConditionalRandomField, self).__init__(**kwargs)
        self.lr_multiplier = lr_multiplier  # 当前层学习率的放大倍数

    @integerize_shape
    def build(self, input_shape):
        super(ConditionalRandomField, self).build(input_shape)
        output_dim = input_shape[-1]
        self._trans = self.add_weight(
            name='trans',
            shape=(output_dim, output_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        if self.lr_multiplier != 1:
            K.set_value(self._trans, K.eval(self._trans) / self.lr_multiplier)

    @property
    def trans(self):
        if self.lr_multiplier != 1:
            return self.lr_multiplier * self._trans
        else:
            return self._trans

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, inputs, mask=None):
        return sequence_masking(inputs, mask, '-inf', 1)

    def target_score(self, y_true, y_pred):
        """计算目标路径的相对概率（还没有归一化）
        要点：逐标签得分，加上转移概率得分。
        """
        point_score = tf.einsum('bni,bni->b', y_true, y_pred)  # 逐标签得分
        trans_score = tf.einsum(
            'bni,ij,bnj->b', y_true[:, :-1], self.trans, y_true[:, 1:]
        )  # 标签转移得分
        return point_score + trans_score

    def log_norm_step(self, inputs, states):
        """递归计算归一化因子
        要点：1、递归计算；2、用logsumexp避免溢出。
        """
        inputs, mask = inputs[:, :-1], inputs[:, -1:]
        states = K.expand_dims(states[0], 2)  # (batch_size, output_dim, 1)
        trans = K.expand_dims(self.trans, 0)  # (1, output_dim, output_dim)
        outputs = tf.reduce_logsumexp(
            states + trans, 1
        )  # (batch_size, output_dim)
        outputs = outputs + inputs
        outputs = mask * outputs + (1 - mask) * states[:, :, 0]
        return outputs, [outputs]

    def dense_loss(self, y_true, y_pred):
        """y_true需要是one hot形式
        """
        # 导出mask并转换数据类型
        mask = K.all(K.greater(y_pred, -1e6), axis=2, keepdims=True)
        mask = K.cast(mask, K.floatx())
        # 计算目标分数
        y_true, y_pred = y_true * mask, y_pred * mask
        target_score = self.target_score(y_true, y_pred)
        # 递归计算log Z
        init_states = [y_pred[:, 0]]
        y_pred = K.concatenate([y_pred, mask], axis=2)
        input_length = K.int_shape(y_pred[:, 1:])[1]
        log_norm, _, _ = K.rnn(
            self.log_norm_step,
            y_pred[:, 1:],
            init_states,
            input_length=input_length
        )  # 最后一步的log Z向量
        log_norm = tf.reduce_logsumexp(log_norm, 1)  # logsumexp得标量
        # 计算损失 -log p
        return log_norm - target_score

    def sparse_loss(self, y_true, y_pred):
        """y_true需要是整数形式（非one hot）
        """
        # y_true需要重新明确一下shape和dtype
        y_true = K.reshape(y_true, K.shape(y_pred)[:-1])
        y_true = K.cast(y_true, 'int32')
        # 转为one hot
        y_true = K.one_hot(y_true, K.shape(self.trans)[0])
        return self.dense_loss(y_true, y_pred)

    def dense_accuracy(self, y_true, y_pred):
        """训练过程中显示逐帧准确率的函数，排除了mask的影响
        此处y_true需要是one hot形式
        """
        y_true = K.argmax(y_true, 2)
        return self.sparse_accuracy(y_true, y_pred)

    def sparse_accuracy(self, y_true, y_pred):
        """训练过程中显示逐帧准确率的函数，排除了mask的影响
        此处y_true需要是整数形式（非one hot）
        """
        # 导出mask并转换数据类型
        mask = K.all(K.greater(y_pred, -1e6), axis=2)
        mask = K.cast(mask, K.floatx())
        # y_true需要重新明确一下shape和dtype
        y_true = K.reshape(y_true, K.shape(y_pred)[:-1])
        y_true = K.cast(y_true, 'int32')
        # 逐标签取最大来粗略评测训练效果
        y_pred = K.cast(K.argmax(y_pred, 2), 'int32')
        isequal = K.cast(K.equal(y_true, y_pred), K.floatx())
        return K.sum(isequal * mask) / K.sum(mask)

    def get_config(self):
        config = {
            'lr_multiplier': self.lr_multiplier,
        }
        base_config = super(ConditionalRandomField, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



def sequence_masking(x, mask, value=0.0, axis=None):
    """为序列条件mask的函数
    mask: 形如(batch_size, seq_len)的0-1矩阵；
    value: mask部分要被替换成的值，可以是'-inf'或'inf'；
    axis: 序列所在轴，默认为1；
    """
    if mask is None:
        return x
    else:
        if K.dtype(mask) != K.dtype(x):
            mask = K.cast(mask, K.dtype(x))
        if value == '-inf':
            value = -1e12
        elif value == 'inf':
            value = 1e12
        if axis is None:
            axis = 1
        elif axis < 0:
            axis = K.ndim(x) + axis
        assert axis > 0, 'axis must be greater than 0'
        for _ in range(axis - 1):
            mask = K.expand_dims(mask, 1)
        for _ in range(K.ndim(x) - K.ndim(mask)):
            mask = K.expand_dims(mask, K.ndim(mask))
        return x * mask + value * (1 - mask)


def integerize_shape(func):
    """装饰器，保证input_shape一定是int或None
    """
    def convert(item):
        if hasattr(item, '__iter__'):
            return [convert(i) for i in item]
        elif hasattr(item, 'value'):
            return item.value
        else:
            return item

    def new_func(self, input_shape):
        input_shape = convert(input_shape)
        return func(self, input_shape)

    return new_func