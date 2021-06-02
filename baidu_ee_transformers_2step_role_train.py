# -*- coding: utf-8 -*-
"""

-------------------------------------------------
   File Name：        ee_transformers_2step_role
   Description :
   Author :           heyouxin
   Create date：      2021/5/26
   Latest version:    v1.0.0
-------------------------------------------------
   Change Log:
#-----------------------------------------------#
    v1.0.0            hyx        2021/5/26
    1.
#-----------------------------------------------#
-------------------------------------------------

"""
from torchcrf import CRF
from torch import nn
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

def load_data_a(filename):
	D = []
	with open(filename) as f:
		for l in f:
			l = json.loads(l)

			for event in l['event_list']:
				arguments = {}
				for argument in event['arguments']:
					key = argument['argument']
					value = (event['event_type'], argument['role'],argument['argument_start_index'])
					arguments[key] = value
				D.append((l['text'], arguments,event['trigger'],event['trigger_start_index']))
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
	for text, arguments,t,t_i in tqdm(data):
		#text, arguments,t,t_i = valid_data_a[0]
		inv_arguments = {(v[0],v[1]): k for k, v in arguments.items()}

		pred_arguments = extract_arguments(text,t,t_i)

		pred_inv_arguments = {(v[0],v[1]): k for k, v in pred_arguments.items()}
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
	trigger_idx = []
	for d in tqdm(train_data):
		#d = train_data_a[0]
		text = d[0]
		text_tok = tokenizer([text],return_tensors='pt',is_split_into_words=True,return_offsets_mapping=True)
		off_map = text_tok['offset_mapping'][0]
		token_ids = text_tok.input_ids[0]
		labels =[-100] + [0 for _ in range(len(token_ids)-2)]+[-100]
		arguments = d[1]
		for argument in arguments.items():

			a_token_ids = tokenizer.encode(argument[0])[1:-1]
			for i,o_m in enumerate(off_map):
				if o_m[0] == argument[1][2] and o_m[1] != 0:
					start_index = i
					labels[start_index] = label2id_a[(argument[1][0],argument[1][1])] * 2 + 1

					for i in range(1, len(a_token_ids)):
						labels[start_index + i] = label2id_a[(argument[1][0],argument[1][1])] * 2 + 2
		t_token_ids = tokenizer.encode(d[2])[1:-1]
		idx = []
		#start_index = 0
		for i, o_m in enumerate(off_map):
			if o_m[0] == d[3] and o_m[1] != 0 and i<128:
				#start_index = i
				idx.append(i)
				if i+len(t_token_ids)-1 < 128:
					idx.append(i+len(t_token_ids)-1)
				else:
					idx.append(127)


				# for j in range(1, len(t_token_ids)):
				# 	idx.append(i + j)
		if not idx:
			idx.append(0)
			idx.append(0)
		trigger_idx.append(idx)
		x.append(text)
		y.append(labels)
	return x,y,trigger_idx
'''7, 11, 25, 53,74,97,101,114,127
i = 130
train_data_a[i]
extract_arguments(x_a[i],train_data_a[i][2],train_data_a[i][3])

'''
def extract_arguments(text,t,t_i):
	"""arguments抽取函数
	"""
	#text = x_a[0]
	#t= train_data_a[0][2]
	#t_i = train_data_a[0][3]
	#t_i = torch.tensor([trigger_idx[0]]).to(device)
	#y_a[130]
	#text = '前两天，被称为 “ 仅次于苹果的软件服务商 ” 的Oracle（ 甲骨文 ）公司突然宣布在中国裁员。。'
	#t_i = 48
	#x_tok = tokenizer([text], max_length=128, padding='max_length',is_split_into_words = True,return_offsets_mapping=True,return_tensors='pt')
	x_tok = tokenizer([text],  is_split_into_words=True,return_offsets_mapping=True, return_tensors='pt')

	mapping = x_tok.offset_mapping[0].numpy()
	x_tok.pop('offset_mapping')
	x_tok.to(device)
	# start_index = 0
	# for i, o_m in enumerate(mapping):
	# 	if o_m[0] == t_i and o_m[1] != 0 and i < 128:
	# 		start_index = i
	t_token_ids = tokenizer.encode(t)[1:-1]
	idx = []
	# start_index = 0
	for i, o_m in enumerate(mapping):
		if o_m[0] == t_i and o_m[1] != 0 and i < 128:
			# start_index = i
			idx.append(i)
			if i + len(t_token_ids) - 1 < 128:
				idx.append(i + len(t_token_ids) - 1)
			else:
				idx.append(127)

	# for j in range(1, len(t_token_ids)):
	# 	idx.append(i + j)
	if not idx:
		idx.append(0)
		idx.append(0)
	input_ids = x_tok['input_ids'].to(device)
	attention_mask = x_tok['attention_mask'].to(device)
	token_type_ids = x_tok['token_type_ids'].to(device)
	#t_i = torch.tensor([[start_index]]).to(device)
	t_i = torch.tensor([idx]).to(device)
	model_a.eval()
	# without crf
	output = model_a(input_ids,attention_mask,token_type_ids,t_i)
	labels = torch.argmax(output,axis=-1)[0].cpu().numpy()
	# #crf decode
	#output,loss = model_a(input_ids, attention_mask, token_type_ids, t_i)
	#labels = output[0]

	#labels = np.argmax(output, axis=-1)[0]
	arguments, starting = [], False
	for i, label in enumerate(labels):
		if label > 0:
			if label % 2 == 1:
				starting = True
				arguments.append([[i], id2label_a[(label - 1) // 2]])
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
	train_data_a = load_data_a('./datasets/DuEE/train.json')
	valid_data_a = load_data_a('./datasets/DuEE/dev.json')



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
	tokenizer = BertTokenizerFast.from_pretrained(model_dir, model_max_length=128, max_length=128, padding_side='right',
	                                              padding='max_length', truncation='True',return_tensors='pt',is_split_into_words=True,return_offsets_mapping=True)


	x_a, y_a,trigger_idx = general_y_label(train_data_a)


	train_x_a = tokenizer(x_a, padding=True, truncation=True, return_tensors='pt')
	train_y_a = pad_sequence([torch.from_numpy(np.array(x[:128])) for x in y_a], batch_first=True)

	class Dataset_a(torch.utils.data.Dataset):
		def __init__(self, encodings, labels,tri):
			self.encodings = encodings
			self.labels = labels
			self.tri = tri

		def __getitem__(self, idx):
			item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
			item['labels'] = torch.tensor(self.labels[idx])
			item['trigger_idx'] = torch.tensor(self.tri[idx])
			return item

		def __len__(self):
			return len(self.labels)


	train_dataset_a = Dataset_a(train_x_a, train_y_a,trigger_idx)
	train_loader_a = DataLoader(train_dataset_a, batch_size=12, shuffle=True)



	class ConditionalLayerNorm(nn.Module):
		def __init__(self,
		             normalized_shape,
		             eps=1e-12):
			super().__init__()

			self.eps = eps

			self.weight = nn.Parameter(torch.Tensor(normalized_shape))
			self.bias = nn.Parameter(torch.Tensor(normalized_shape))
			#
			self.weight_dense = nn.Linear(normalized_shape * 2, normalized_shape, bias=False)
			self.bias_dense = nn.Linear(normalized_shape * 2, normalized_shape, bias=False)

			# self.weight_dense = nn.Linear(normalized_shape , normalized_shape, bias=False)
			# self.bias_dense = nn.Linear(normalized_shape , normalized_shape, bias=False)

			self.reset_weight_and_bias()

		def reset_weight_and_bias(self):
			"""
			此处初始化的作用是在训练开始阶段不让 conditional layer norm 起作用
			"""
			nn.init.ones_(self.weight)
			nn.init.zeros_(self.bias)

			nn.init.zeros_(self.weight_dense.weight)
			nn.init.zeros_(self.bias_dense.weight)

		def forward(self, inputs, cond=None):
			assert cond is not None, 'Conditional tensor need to input when use conditional layer norm'
			cond = torch.unsqueeze(cond, 1)  # (b, 1, h*2)

			weight = self.weight_dense(cond) + self.weight  # (b, 1, h)
			bias = self.bias_dense(cond) + self.bias  # (b, 1, h)

			mean = torch.mean(inputs, dim=-1, keepdim=True)  # （b, s, 1）
			outputs = inputs - mean  # (b, s, h)

			variance = torch.mean(outputs ** 2, dim=-1, keepdim=True)
			std = torch.sqrt(variance + self.eps)  # (b, s, 1)

			outputs = outputs / std  # (b, s, h)

			outputs = outputs * weight + bias

			return outputs


	# role 提取器
	class RoleExtractor(torch.nn.Module):
		def __init__(self,model_dir,dropout_prob,num_labels):
			#,model_dir,dropout_prob,num_labels
			super(RoleExtractor, self).__init__()
			self.l1 = BertModel.from_pretrained(model_dir)
			self.bert_config = self.l1.config
			out_dims = self.bert_config.hidden_size
			self.num_labels = num_labels
			self.dropout = torch.nn.Dropout(dropout_prob)
			self.classifier = torch.nn.Linear(out_dims, self.num_labels)
			self.conditional_layer_norm = ConditionalLayerNorm(768, eps=self.bert_config.layer_norm_eps)
			#self.crf_module = CRF(num_tags=num_labels, batch_first=True)
		# self._init_weights(init_blocks, initializer_range=self.bert_config.initializer_range)

		def forward(self,
		            token_ids,
		            attention_mask,
		            token_type_ids,
		            tri_idx,
		            labels=None):

			outputs = self.l1(
				input_ids=token_ids,
				attention_mask=attention_mask,
				token_type_ids=token_type_ids
			)
			# [12,128,768]  [12,768]
			seq_out, pooled_out = outputs[0], outputs[1]
			# [12,1,768]
			trigger_label_feature = self._batch_gather(seq_out, tri_idx)
			# [12,768]
			trigger_label_feature = trigger_label_feature.view([trigger_label_feature.size()[0], -1])

			seq_out = self.conditional_layer_norm(seq_out, trigger_label_feature)


			seq_out = self.dropout(seq_out)
			logits = self.classifier(seq_out)

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
			'''
			if labels is not None:
				tokens_loss = -1. * self.crf_module(emissions=logits,
				                                    tags=labels.long(),
				                                    mask=attention_mask.byte(),
				                                    reduction='mean')

				return logits,tokens_loss

			else:
				tokens_out = self.crf_module.decode(emissions=logits, mask=attention_mask.byte())
				return tokens_out,loss
			'''

		def _batch_gather(self,data: torch.Tensor, index: torch.Tensor):
			"""
			实现类似 tf.batch_gather 的效果
			:param data: (bs, max_seq_len, hidden)
			:param index: (bs, n)
			:return: a tensor which shape is (bs, n, hidden)
			"""

			index = index.unsqueeze(-1).repeat_interleave(data.size()[-1], dim=-1)  # (bs, n, hidden)

			return torch.gather(data, 1, index)


	model_a = RoleExtractor(model_dir, 0.1, num_labels_a)
	model_a.to(device)

	best_val_f1 = 0
	model_a.train()
	optim = torch.optim.Adam(model_a.parameters(), lr=5e-10)
	for epoch in range(1):
		for _, batch in tqdm(enumerate(train_loader_a)):

			input_ids = batch['input_ids'].to(device)
			attention_mask = batch['attention_mask'].to(device)
			token_type_ids = batch['token_type_ids'].to(device)
			labels = batch['labels'].to(device)
			tri_idx = batch['trigger_idx'].to(device)
			'''
			b_m = BertModel.from_pretrained(model_dir).to(device)
			b_o = b_m(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)

			seq_out, pooled_out = b_o[0], b_o[1]
			tri_idx = torch.tensor([[ 6,7,8],[12],[12],[20],[48],[25],[16]]
			trigger_label_feature = _batch_gather(seq_out, tri_idx)

			trigger_label_feature = trigger_label_feature.view([trigger_label_feature.size()[0], -1])

			conditional_layer_norm = ConditionalLayerNorm(768, eps=b_m.config.layer_norm_eps).to(device)

			seq_out = conditional_layer_norm(seq_out, trigger_label_feature)
			cond = torch.unsqueeze(trigger_label_feature, 1)
			weight_dense = nn.Linear(768 , 128, bias=False).to(device)
			weight = weight_dense(cond)
			'''



			outputs, loss = model_a(input_ids, attention_mask, token_type_ids,tri_idx, labels)
			if _ % 100 == 0:
				print(f'Epoch: {epoch}, step:{_}, Loss:  {loss.item()}')

			optim.zero_grad()
			loss.backward()
			optim.step()


		f1, precision, recall = evaluate(valid_data_a)
		if f1 >= best_val_f1:
			best_val_f1 = f1
			torch.save(model_a.state_dict(), 'pt_checkpoint/model_a_weights.pth')
		print(
			'f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
			(f1, precision, recall, best_val_f1)
		)


	torch.save(model_a.state_dict(), 'pt_checkpoint/model_a_2_weights.pth')

	model_a.load_state_dict(torch.load('pt_checkpoint/model_a_weights.pth'))

	_batch_gather(torch.tensor([[1,2,3],[2,3,4]]),torch.tensor([0,1]))
	def _batch_gather(data: torch.Tensor, index: torch.Tensor):
		"""
		实现类似 tf.batch_gather 的效果
		:param data: (bs, max_seq_len, hidden)
		:param index: (bs, n)
		:return: a tensor which shape is (bs, n, hidden)
		"""
		index = index.unsqueeze(-1).repeat_interleave(data.size()[-1], dim=-1)  # (bs, n, hidden)
		return torch.gather(data, 1, index)



	for i in range(1,len(train_data_a)):
		if train_data_a[i][0] == train_data_a[i-1][0]:
			print(i)