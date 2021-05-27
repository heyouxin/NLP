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

def load_data_t(filename):
	D = []
	with open(filename) as f:
		for l in f:

			l = json.loads(l)
			arguments = {}
			for event in l['event_list']:
				key = event['trigger']
				value = (event['event_type'], 'trigger',event['trigger_start_index'])
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
	#data = train_data_t[128:129]
	for text, arguments in tqdm(data):

		inv_arguments = {v[0]: k for k, v in arguments.items()}
		pred_arguments = extract_arguments(text)
		pred_inv_arguments = {v[0]: k for k, v in pred_arguments.items()}
		#pred_inv_arguments = pred_arguments
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


def general_y_label(train_data):
	y = []
	x = []
	trigger_index = []
	for d in tqdm(train_data):
		#d = train_data_t[128]
		text = d[0]
		text_tok = tokenizer([text],return_tensors='pt',is_split_into_words=True,return_offsets_mapping=True)
		off_map = text_tok['offset_mapping'][0]
		token_ids = text_tok.input_ids[0]
		labels =[-100] + [0 for _ in range(len(token_ids)-2)]+[-100]
		idx = []
		arguments = d[1]
		for argument in arguments.items():
			idx_1 = []
			a_token_ids = tokenizer.encode(argument[0])[1:-1]
			for i,o_m in enumerate(off_map):
				if o_m[0] == argument[1][2] and o_m[1] != 0:
					start_index = i
					labels[start_index] = label2id_t[argument[1][0]] * 2 + 1
					idx_1.append(start_index)
					for i in range(1, len(a_token_ids)):
						labels[start_index + i] = label2id_t[argument[1][0]] * 2 + 2
						idx_1.append(start_index + i)
			idx.append(idx_1)
		trigger_index.append(idx)
		x.append(text)
		y.append(labels)
	return x,y,trigger_index



def extract_arguments(text):
	"""arguments抽取函数
	"""
	#text = x_t[128]
	x_tok = tokenizer([text],is_split_into_words = True,return_offsets_mapping=True,return_tensors='pt')
	mapping = x_tok.offset_mapping[0].numpy()
	x_tok.pop('offset_mapping')
	x_tok.to(device)
	input_ids = x_tok['input_ids'].to(device)
	attention_mask = x_tok['attention_mask'].to(device)
	token_type_ids = x_tok['token_type_ids'].to(device)
	model_t.eval()
	output = model_t(input_ids,attention_mask,token_type_ids)
	labels = torch.argmax(output,axis=-1)[0].cpu().numpy()
	#labels = np.argmax(output, axis=-1)[0]
	arguments, starting = [], False
	for i, label in enumerate(labels):
		if label > 0:
			if label % 2 == 1:
				starting = True
				arguments.append([[i], id2label_t[(label - 1) // 2]])
			elif starting:
				arguments[-1][0].append(i)
			else:
				starting = False
		else:
			starting = False

	return {
		text[mapping[w[0]][0]:mapping[w[-1]][-1]]: (l,w)
		for w, l in arguments
	}



if __name__ == '__main__':
	seq_len = 156
	# 读取数据
	train_data_t = load_data_t('./datasets/DuEE/train.json')
	valid_data_t = load_data_t('./datasets/DuEE/dev.json')

	# 读取schema
	with open('./datasets/DuEE/event_schema.json') as f:
		id2label_t, label2id_t, n = {}, {}, 0
		for l in f:
			l = json.loads(l)

			key = (l['event_type'])
			id2label_t[n] = key
			label2id_t[key] = n
			n += 1
		num_labels_t = len(id2label_t) * 2 + 1


	model_dir = '../pretrain_model/bert-base-chinese'
	tokenizer = BertTokenizerFast.from_pretrained(model_dir, model_max_length=seq_len, max_length=seq_len, padding_side='right',
	                                              padding='max_length', truncation='True',return_tensors='pt',is_split_into_words=True,return_offsets_mapping=True)

	x_tok = tokenizer([train_data_t[0][0]],return_tensors='pt',is_split_into_words=True,return_offsets_mapping=True)

	x_t, y_t,trigger_index = general_y_label(train_data_t)


	train_x_t = tokenizer(x_t, padding=True, truncation=True, return_tensors='pt')
	train_y_t = pad_sequence([torch.from_numpy(np.array(x[:seq_len])) for x in y_t], batch_first=True)




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


	# trigger 提取器
	class TriggerExtractor(torch.nn.Module):
		def __init__(self,model_dir,dropout_prob,num_labels):
			#,model_dir,dropout_prob,num_labels
			super(TriggerExtractor, self).__init__()
			self.l1 = BertModel.from_pretrained(model_dir)
			self.bert_config = self.l1.config
			out_dims = self.bert_config.hidden_size
			self.num_labels = num_labels
			self.dropout = torch.nn.Dropout(dropout_prob)
			self.classifier = torch.nn.Linear(out_dims, self.num_labels)

		# self._init_weights(init_blocks, initializer_range=self.bert_config.initializer_range)

		def forward(self,
		            token_ids,
		            attention_mask,
		            token_type_ids,
		            labels=None):

			outputs = self.l1(
				input_ids=token_ids,
				attention_mask=attention_mask,
				token_type_ids=token_type_ids
			)

			sequence_output = outputs[0]

			sequence_output = self.dropout(sequence_output)
			logits = self.classifier(sequence_output)

			loss = None
			if labels is not None:
				loss_fct = torch.nn.CrossEntropyLoss()
				# Only keep active parts of the loss
				if attention_mask is not None:
					active_loss = attention_mask.view(-1) == 1
					active_logits = logits.view(-1, self.num_labels)
					active_labels = torch.where(
						active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
					)
					loss = loss_fct(active_logits, active_labels)
				else:
					loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
				return logits,loss
			else:
				return logits


	model_t = TriggerExtractor(model_dir,0.1,num_labels_t)
	model_t.to(device)

	best_val_f1 = 0
	model_t.train()
	optim = torch.optim.Adam(model_t.parameters(), lr=2e-5)
	for epoch in range(1):
		for _, batch in tqdm(enumerate(train_loader_t)):

			input_ids = batch['input_ids'].to(device)
			attention_mask = batch['attention_mask'].to(device)
			token_type_ids = batch['token_type_ids'].to(device)
			labels = batch['labels'].to(device)
			outputs,loss = model_t(input_ids, attention_mask, token_type_ids, labels)
			if _ % 100 == 0:
				print(f'Epoch: {epoch}, step:{_}, Loss:  {loss.item()}')

			optim.zero_grad()
			loss.backward()
			optim.step()

		f1, precision, recall = evaluate(valid_data_t)
		if f1 >= best_val_f1:
			best_val_f1 = f1
			torch.save(model_t.state_dict(), 'pt_checkpoint/model_t_weights.pth')
		print(
			'f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
			(f1, precision, recall, best_val_f1)
		)


	arguments_t = extract_arguments(x_t[128])
	print(arguments_t)


	torch.save(model_t.state_dict(), 'pt_checkpoint/model_t_weights.pth')

	model_t.load_state_dict(torch.load('pt_checkpoint/model_t_weights.pth'))

	from torchcrf import CRF

	torch.cuda.empty_cache()
	f1, precision, recall = evaluate(valid_data_t)
	print(
		'f1: %.5f, precision: %.5f, recall: %.5f' %
		(f1, precision, recall)
	)








	for i,c in enumerate(train_data_t):
		if len(c[1]) > 2:
			print(i)