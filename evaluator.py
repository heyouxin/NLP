# -*- coding: utf-8 -*-
"""

-------------------------------------------------
   File Name：        evaluator
   Description :
   Author :           heyouxin
   Create date：      2021/5/25
   Latest version:    v1.0.0
-------------------------------------------------
   Change Log:
#-----------------------------------------------#
    v1.0.0            hyx        2021/5/25
    1.
#-----------------------------------------------#
-------------------------------------------------

"""
from tqdm import tqdm
import pylcs
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
import torch
def extract_arguments(text,model,tokenizer,id2label):
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
	labels = torch.argmax(output[0],axis=-1)[0].cpu().numpy()
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

def evaluate(data,model,tokenizer,id2label):
	"""评测函数（跟官方评测结果不一定相同，但很接近）
	"""
	X, Y, Z = 1e-10, 1e-10, 1e-10
	for text, arguments in tqdm(data):
		inv_arguments = {v: k for k, v in arguments.items()}
		pred_arguments = extract_arguments(text,model,tokenizer,id2label)
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


