# -*- coding: utf-8 -*-
"""

-------------------------------------------------
   File Name：        baidu_ee_pytorch
   Description :
   Author :           heyouxin
   Create date：      2021/5/19
   Latest version:    v1.0.0
-------------------------------------------------
   Change Log:
#-----------------------------------------------#
    v1.0.0            hyx        2021/5/19
    1.
#-----------------------------------------------#
-------------------------------------------------

"""

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig,BertForTokenClassification,BertTokenizerFast
import json
import pylcs
from tqdm import tqdm
import numpy as np

from torch.nn.utils.rnn import pad_sequence

# # Setting up the device for GPU usage
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'


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
	#text = x[0]
	x_tok = tokenizer([text],is_split_into_words = True,return_offsets_mapping=True,return_tensors='pt')
	mapping = x_tok.offset_mapping[0].numpy()
	x_tok.pop('offset_mapping')
	x_tok.to(device)
	input_ids = x_tok['input_ids'].to(device)
	attention_mask = x_tok['attention_mask'].to(device)
	token_type_ids = x_tok['token_type_ids'].to(device)
	model.eval()
	output = model(input_ids,attention_mask,token_type_ids)
	labels = torch.argmax(output['logits'],axis=-1)[0].cpu().numpy()
	#labels = np.argmax(output, axis=-1)[0]
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

	model_dir = '../pretrain_model/cn_bert_wwm_pytorch'
	tokenizer = BertTokenizerFast.from_pretrained(model_dir, model_max_length=128, max_length=128, padding_side='right',
	                                              padding='max_length', truncation='True')

	x, y = general_y_label(train_data, tokenizer)
	train_x = tokenizer(x, padding=True, truncation=True, return_tensors='pt')



	train_y = pad_sequence([torch.from_numpy(np.array(x[:128])) for x in y], batch_first=True)

	class MyDataset(torch.utils.data.Dataset):
		def __init__(self, encodings, labels):
			self.encodings = encodings
			self.labels = labels

		def __getitem__(self, idx):
			item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
			item['labels'] = torch.tensor(self.labels[idx])
			return item

		def __len__(self):
			return len(self.labels)


	train_dataset = MyDataset(train_x, train_y)
	train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True)

	class BERTClass(torch.nn.Module):
		def __init__(self):
			super(BERTClass, self).__init__()
			self.l1 = BertForTokenClassification.from_pretrained(model_dir,num_labels=num_labels)
			#self.l2 = torch.nn.Linear(768, num_labels)


		def forward(self,input_ids, attention_mask,token_type_ids,labels=None):
			output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask,token_type_ids=token_type_ids,labels=labels)
			#output = self.l2(output_1)
			return output_1

	model = BERTClass()
	model.to(device)
	best_val_f1 = 0



	model.train()
	optim = torch.optim.Adam(model.parameters(),lr = 5e-7)
	for epoch in range(1):
		for _,batch in tqdm(enumerate(train_loader)):

			input_ids = batch['input_ids'].to(device)
			attention_mask = batch['attention_mask'].to(device)
			token_type_ids = batch['token_type_ids'].to(device)
			labels = batch['labels'].to(device)
			outputs = model(input_ids, attention_mask,token_type_ids, labels)
			loss = outputs.loss

			if _ % 100 == 0:
				print(f'Epoch: {epoch}, step:{_}, Loss:  {loss.item()}')

			optim.zero_grad()
			loss.backward()
			optim.step()

		f1, precision, recall = evaluate(valid_data)
		if f1 >= best_val_f1:
			best_val_f1 = f1
			torch.save(model.state_dict(), 'pt_checkpoint/model_weights.pth')
		print(
			'f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
			(f1, precision, recall, best_val_f1)
		)






	#inference
	model = BERTClass()
	model.load_state_dict(torch.load('pt_checkpoint/model_weights.pth'))
	model.to(device)
	model.eval()
	extract_arguments(x[1])


	f1, precision, recall = evaluate(valid_data)
	print(
		'f1: %.5f, precision: %.5f, recall: %.5f' %
		(f1, precision, recall)
	)










	import numpy as np
	from datasets import load_metric

	metric = load_metric("accuracy")


	def compute_metrics(eval_pred):
		logits, labels = eval_pred
		predictions = np.argmax(logits, axis=-1)
		return metric.compute(predictions=predictions, references=labels)


