# -*- coding: utf-8 -*-
"""

-------------------------------------------------
   File Name：        baidu_ee_transformers_2step
   Description :
   Author :           heyouxin
   Create date：      2021/5/16
   Latest version:    v1.0.0
-------------------------------------------------
   Change Log:
#-----------------------------------------------#
    v1.0.0            hyx        2021/5/16
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

def load_data_t(filename):
	D = []
	with open(filename) as f:
		for l in f:
			l = json.loads(l)
			arguments = {}
			for event in l['event_list']:
				key = event['trigger']
				value = (event['event_type'], 'trigger')
				arguments[key] = value
			D.append((l['text'], arguments))
	return D

def load_data_a(filename):
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


def general_y_label(train_data,tokenizer,label2id,id2label):
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

	model_t.eval()
	model_a.eval()
	#text = x_a[0]
	x_tok = tokenizer([text],is_split_into_words = True,return_offsets_mapping=True,return_tensors='pt')
	mapping = x_tok.offset_mapping[0].numpy()
	x_tok.pop('offset_mapping')
	x_tok.to(device)
	input_ids = x_tok['input_ids'].to(device)
	attention_mask = x_tok['attention_mask'].to(device)
	token_type_ids = x_tok['token_type_ids'].to(device)
	#labels_t = x_tok['labels'].to(device)

	output = model_t(input_ids,attention_mask,token_type_ids)
	logits_t = output['logits']
	arguments_t = extract(logits_t, text, mapping, id2label_t)
	#labels = torch.argmax(output['logits'],axis=-1)[0].cpu().numpy()
	#labels = np.argmax(output, axis=-1)[0]
	# arguments, starting = [], False
	# for i, label in enumerate(labels):
	# 	if label > 0:
	# 		if label % 2 == 1:
	# 			starting = True
	# 			arguments.append([[i], id2label_t[(label - 1) // 2]])
	# 		elif starting:
	# 			arguments[-1][0].append(i)
	# 		else:
	# 			starting = False
	# 	else:
	# 		starting = False
	# arguments_t = {
	# 	text[mapping[w[0]][0]:mapping[w[-1]][-1]]: l
	# 	for w, l in arguments
	# }

	#a_input = get_a_input(output)
	logits_a = model_a(logits_t)
	arguments_a = extract(logits_a, text, mapping,id2label_a)


	return arguments_t,arguments_a


def extract(logits_a,text,mapping,id2label):
	#id2label = id2label_a
	labels = torch.argmax(logits_a, axis=-1)[0].cpu().numpy()
	# labels = np.argmax(output, axis=-1)[0]
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
	try:
		return {
			text[mapping[w[0]][0]:mapping[w[-1]][-1]]: l
			for w, l in arguments
		}
	except:
		return {}



if __name__ == '__main__':

	# 读取数据
	train_data_t = load_data_t('./datasets/DuEE/train.json')
	train_data_a = load_data_a('./datasets/DuEE/train.json')

	#valid_data = load_data('./datasets/DuEE/dev.json')

	# 读取schema
	with open('./datasets/DuEE/event_schema.json') as f:
		id2label_t, label2id_t, n = {}, {}, 0
		for l in f:
			l = json.loads(l)

			key = (l['event_type'], 'trigger')
			id2label_t[n] = key
			label2id_t[key] = n
			n += 1
		num_labels_t = len(id2label_t) * 2 + 1


	# 读取schema
	with open('./datasets/DuEE/event_schema.json') as f:
		id2label_a, label2id_a, n = {}, {}, 0
		for l in f:
			l = json.loads(l)
			for role in l['role_list']:
				key = (l['event_type'], role['role'])
				id2label_a[n] = key
				label2id_a[key] = n
				n += 1
		num_labels_a = len(id2label_a) * 2 + 1


	model_dir = '../pretrain_model/bert-base-chinese'
	tokenizer = BertTokenizerFast.from_pretrained(model_dir,model_max_length =128,max_length = 128,padding_side = 'right',padding='max_length',truncation='True')

	x_t,y_t = general_y_label(train_data_t,tokenizer,label2id_t,id2label_t)
	x_a, y_a = general_y_label(train_data_a, tokenizer,label2id_a,id2label_a)

	train_x_t = tokenizer(x_t, padding=True, truncation=True, return_tensors='pt')
	train_y_t = pad_sequence([torch.from_numpy(np.array(x[:128])) for x in y_t], batch_first=True)

	train_x_a = tokenizer(x_a, padding=True, truncation=True, return_tensors='pt')
	train_y_a = pad_sequence([torch.from_numpy(np.array(x[:128])) for x in y_a], batch_first=True)


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


	train_dataset_t = MyDataset(train_x_t, train_y_t)
	train_loader_t = DataLoader(train_dataset_t, batch_size=12, shuffle=True)

	train_dataset_a = MyDataset(train_x_a, train_y_a)
	train_loader_a = DataLoader(train_dataset_a, batch_size=12, shuffle=True)


	class BERTClass_t(torch.nn.Module):
		def __init__(self):
			super(BERTClass_t, self).__init__()
			self.l1 = BertForTokenClassification.from_pretrained(model_dir,num_labels=num_labels_t)
			#self.l2 = torch.nn.Linear(768, num_labels)


		def forward(self,input_ids, attention_mask,token_type_ids,labels=None):
			output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask,token_type_ids=token_type_ids,labels=labels)
			#output = self.l2(output_1)
			return output_1


	model_t = BERTClass_t()
	model_t.to(device)

	class BERTClass_a(torch.nn.Module):
		def __init__(self):
			super(BERTClass_a, self).__init__()
			#self.l1 = BertForTokenClassification.from_pretrained(model_dir,num_labels=num_labels_t)
			#self.l1 = torch.nn.lambada
			#m1 = nn.LayerNorm(input.size()[1:])
 			self.l2 = torch.nn.Linear(num_labels_t, num_labels_a)

			self.activation = torch.nn.Softmax()

		def forward(self,x,labels=None):
			#x = torch.nn.LayerNorm(x.size()[1:])
			#x = Lambda:
			a_input = self.get_a_input(x)
			output_1 = self.l2(a_input)

			output_1 = self.activation(output_1)
			#logits = torch.argmax(output_1, -1)

			if labels is not None:
				loss = self.compute_loss_a(output_1,labels)
				return output_1, loss
			else:
				return output_1

		def compute_loss_a(self,pred, labels):
			crossentropyloss = torch.nn.CrossEntropyLoss()
			loss_a =0
			for p,l in zip(pred,labels):
				loss_a += crossentropyloss(p, l)
			return loss_a

		def get_a_input(self,output):
			# loss_t = output.loss
			#logits_t = output.logits
			t_seq = torch.argmax(output, axis=-1)
			a_input = torch.Tensor().to(device)
			for t, l in zip(t_seq, output):
				s_e = torch.where(t > 0)[0].cpu().numpy()
				if len(s_e) >= 2:
					t_t = l[torch.where(t > 0)[0].cpu().numpy()[0]:torch.where(t > 0)[0].cpu().numpy()[1] + 1]
					t_l = torch.cat([t_t, l], 0)
				else:
					t_t = torch.zeros([2, 131]).to(device)
					t_l = torch.cat([l, t_t], 0)

				a_input = torch.cat([a_input, t_l[:128, :]], 0)
			a_input = a_input.reshape([output.size()[0], -1, output.size()[2]]).to(device)
			return a_input
	model_a = BERTClass_a()
	model_a.to(device)


	model_t.train()
	model_a.train()
	from itertools import chain
	optim = torch.optim.Adam(chain(model_t.parameters(),model_a.parameters()), lr=5e-5)
	#optim = torch.optim.Adam(model_a.parameters(), lr=2e-5)
	_ = 0


	for epoch in range(1):
		for batch,batch_a in tqdm(zip(train_loader_t,train_loader_a)):
			_ += 1
			input_ids = batch['input_ids'].to(device)
			attention_mask = batch['attention_mask'].to(device)
			token_type_ids = batch['token_type_ids'].to(device)
			labels = batch['labels'].to(device)
			outputs = model_t(input_ids, attention_mask, token_type_ids, labels)
			loss_t = outputs.loss
			#logits_t = outputs.logits
			#a_input = get_a_input(outputs)
			labels_a = batch_a['labels'].to(device)
			logits_a,loss_a = model_a(outputs.logits,labels_a)
			loss = loss_t + loss_a
			if _ % 100 == 0:
				print(f'Epoch: {epoch}, step:{_}, Loss:  {loss.item()}')

			optim.zero_grad()
			loss.backward()
			optim.step()


	arguments_t,arguments_a = extract_arguments(x_t[1379])
	print(arguments_t)
	print(arguments_a)


	torch.save(model_t.state_dict(), 'pt_checkpoint/model_t_weights.pth')
	torch.save(model_a.state_dict(), 'pt_checkpoint/model_a_weights.pth')

	model_t.load_state_dict(torch.load('pt_checkpoint/model_t_weights.pth'))
	model_a.load_state_dict(torch.load('pt_checkpoint/model_a_weights.pth'))

	from torchcrf import CRF
	torch.cuda.empty_cache()