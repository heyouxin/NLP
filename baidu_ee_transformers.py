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
import tensorflow_addons as tfa
import tensorflow as tf
from tensorflow.keras import Model
import json
import pylcs
from tqdm import tqdm
import numpy as np
from transformers import BertTokenizerFast, AutoTokenizer, BertTokenizer, TFBertForSequenceClassification, \
	TFBertForTokenClassification, TFAlbertModel
from tensorflow.keras.layers import Dense
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
	#text = x[10605]
	x_tok = tokenizer([text],is_split_into_words = True,return_offsets_mapping=True,return_tensors='tf')
	mapping = x_tok.offset_mapping[0].numpy()
	x_tok.pop('offset_mapping')

	output = model(x_tok)
	#labels =np.argmax(output['logits'],axis=-1)[0]
	labels = np.argmax(output, axis=-1)[0]
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
		text[mapping[w[0]][0]:mapping[w[-1]][-1]]: l
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


	model_dir = '../pretrain_model/chinese-bert-wwm-ext'
	tokenizer = BertTokenizerFast.from_pretrained(model_dir,model_max_length =128,max_length = 128,padding_side = 'right',padding='max_length',truncation='True')

	x,y = general_y_label(train_data,tokenizer)
	train_x = tokenizer(x, padding=True, truncation=True, return_tensors='tf')
	train_y = tf.keras.preprocessing.sequence.pad_sequences(y,maxlen=128,padding='post',truncating='post')
	#train_y = tf.keras.preprocessing.sequence.pad_sequences(train_y, maxlen=128, padding='post',
	#                                                        truncating='post')

	train_ds = tf.data.Dataset.from_tensor_slices(((dict(train_x), train_y))).shuffle(100).batch(12)

	'''
	class MyModel(Model):
		def __init__(self):
			super(MyModel, self).__init__()
			self.d1 = TFBertForTokenClassification.from_pretrained(model_dir, num_labels=num_labels)
			#self.d2 = Dense(num_labels, activation='softmax')
			#self.d2 = Dense(num_labels, activation='softmax')
			self.transition_params = tf.Variable(tf.random.uniform(shape=(num_labels, num_labels)))

		def call(self, x,labels=None):
			text_lens = tf.math.reduce_sum(tf.cast(tf.math.not_equal(x['attention_mask'], 0), dtype=tf.int32), axis=-1)
			#text_lens = tf.math.reduce_sum(tf.cast(tf.math.not_equal(x, 0), dtype=tf.int32), axis=-1)
			#text_lens = tf.constant([128 for _ in range(labels.shape[0])])
			x1 = self.d1(x)
			#logits = self.d2(x1.logits)
			logits = x1.logits
			if labels is not None:
				label_sequences = tf.convert_to_tensor(labels, dtype=tf.int32)
				log_likelihood, self.transition_params = tfa.text.crf_log_likelihood(logits,
				                                                                       label_sequences,
				                                                                       text_lens,
				                                                                  transition_params=self.transition_params)
				return logits,log_likelihood,text_lens
			else:
				return logits,text_lens

	model = MyModel()
	optimizer = tf.keras.optimizers.Adam(5e-5)
	#train_loss = tf.keras.metrics.Mean(name='train_loss')


	@tf.function
	def train_step(train_x, train_y):
		with tf.GradientTape() as tape:

			logits, log_likelihood,text_lens = model(train_x,train_y)
			loss = tf.reduce_mean(-log_likelihood)

		# freeze_conv_var_list = [t for t in model.trainable_variables if
		#                         not t.name.startswith('tf_bert_for_token_classification')]
		# gradients = tape.gradient(loss, freeze_conv_var_list)
		# optimizer.apply_gradients(zip(gradients, freeze_conv_var_list))

		gradients = tape.gradient(loss, model.trainable_variables)
		optimizer.apply_gradients(zip(gradients, model.trainable_variables))

		#train_loss(loss)
		#train_accuracy(train_y, predictions)
		return loss,logits,text_lens

	def get_acc_one_step(logits,labels_batch,text_lens):
		paths = []
		accuracy = 0
		for logit,  labels,text_len in zip(logits, labels_batch,text_lens):
			viterbi_path, _ = tfa.text.viterbi_decode(logit[:text_len], model.transition_params)
			paths.append(viterbi_path)
			correct_prediction = tf.equal(
				tf.convert_to_tensor(tf.keras.preprocessing.sequence.pad_sequences([viterbi_path], padding='post'),
				                     dtype=tf.int32),
				tf.convert_to_tensor(tf.keras.preprocessing.sequence.pad_sequences([labels[:text_len]], padding='post'),
				                     dtype=tf.int32)
			)
			accuracy = accuracy + tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		# print(tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))
		accuracy = accuracy / len(paths)
		return accuracy


	EPOCHS = 1
	for epoch in range(EPOCHS):
		# 在下一个epoch开始时，重置评估指标
		#train_loss.reset_states()

		step = 0
		for x_batch, y_batch in tqdm(train_ds):
			loss,logits,text_lens = train_step(x_batch, y_batch)
			accuracy = get_acc_one_step(logits, y_batch,text_lens)
			if step%100 == 0:
				template = 'step {}, Loss: {}, Acc:{}'
				print(template.format(step,loss,accuracy))
		                      #train_loss.result()))
			step += 1
		template = 'Epoch {}, Loss: {}, Acc:{}'
		print(template.format(epoch + 1, loss, accuracy))



	#inference

	# x_1 = {}
	# x_1['input_ids'] = train_x['input_ids'][0:1]
	# x_1['token_type_ids'] = train_x['token_type_ids'][0:1]
	# x_1['attention_mask'] = train_x['attention_mask'][0:1]

	x_pred = tokenizer([x[1]], padding=True, truncation=True, return_tensors='tf')
	#x_pred['input_ids'] = x_pred['input_ids'][0] + tf.constant([0 for _ in range(104)])
	logits,text_lens = model(x_pred)
	tf.argmax(logits,axis=-1)
	paths = []
	for logit,text_len in zip(logits,text_lens):
		viterbi_path, _ = tfa.text.viterbi_decode(logit[:text_len], model.transition_params)
		paths.append(viterbi_path)
	print(paths[0])

	model.save_weights('./checkpoints/ee_transformers')
	model.load_weights('./checkpoints/ee_transformers')
	'''


	class MyModel(Model):
		def __init__(self):
			super(MyModel, self).__init__()
			self.d1 = TFBertForTokenClassification.from_pretrained(model_dir, num_labels=num_labels)
	

		def call(self, x):
			x1 = self.d1(x)
			return x1.logits

	model = MyModel()
	model.load_weights('./checkpoints/ee_transformers_without_crf')


	loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
	optimizer = tf.keras.optimizers.Adam(5e-10)
	train_loss = tf.keras.metrics.Mean(name='train_loss')
	train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')



	@tf.function
	def train_step(batch_x,batch_y):
		with tf.GradientTape() as tape:
			predictions = model(batch_x)
			loss = loss_object(batch_y, predictions)

		gradients = tape.gradient(loss, model.trainable_variables)
		optimizer.apply_gradients(zip(gradients, model.trainable_variables))
		train_loss.update_state(loss)
		train_accuracy.update_state(batch_y, predictions)

	EPOCHS = 2
	for epoch in range(EPOCHS):


		step = 0
		for batch_x, batch_y in tqdm(train_ds):
			train_step(batch_x,batch_y)
			step += 1
			if step%100 == 0:
				template = 'step: {}, Loss: {}, Acc:{}'
				print(template.format(step,train_loss.result(),train_accuracy.result()*100))

		# 在下一个epoch开始时，重置评估指标
		train_loss.reset_states()
		train_accuracy.reset_states()
		# template = 'Epoch {}, Loss: {}, Accuracy: {}'
		# print (template.format(epoch+1,
		#                      train_loss.result(),train_accuracy.result()*100))




	f1, precision, recall = evaluate(valid_data)
	print(
		'f1: %.5f, precision: %.5f, recall: %.5f' %
		(f1, precision, recall)
	)
	model.save_weights('./checkpoints/ee_transformers_without_crf')
	extract_arguments(x[10000])



	'''
	model = TFBertForTokenClassification.from_pretrained(model_dir,num_labels=num_labels)
	optimizer = tf.keras.optimizers.Adam(5e-5)
	loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
	metrics = tf.keras.metrics.SparseCategoricalAccuracy('sparse_accuracy')
	model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])
	model.summary()

	evaluator = Evaluator()
	history = model.fit(train_ds,epochs=1)
	                    #,callbacks=[evaluator])
	model.save_weights('./checkpoints/ee_transformers_without_crf')
	
	

	# ----inference----
	x_pred = tokenizer(x[10605],return_tensors='tf')
	output = model(x_pred)
	y_label = np.argmax(output,axis=-1)
	extract_arguments(x[3])
	
	'''



