import json
import os
import torch
import random
import copy

def load_dataset(dataset,args):

	def load(path):
		contents=[]
		with open(path,'r') as f:
			for line in f.readlines():
				j=json.loads(line)
				chq=j['chq'].strip()
				faq=j['faq'].strip()
				contents.append((chq,faq,j['peturbations']))
		return contents

	data=load(os.path.join(dataset,'contrast.perturb'))

	return data


class DatasetIterater(object):
	def __init__(self,tokenizer,dataset,batch_size,device,contrast_number,max_src_length):
		self.batch_size=batch_size
		self.dataset=dataset
		self.device=device
		self.tokenizer=tokenizer
		self.n_batches=(len(dataset)-1) //batch_size
		self.res=False
		if len(dataset)% self.n_batches!=0:
			self.res=True
		self.index=0
		self.contrast_number=contrast_number
		self.max_src_length=max_src_length
	

	def text_to_tensor(self,texts):
		o=self.tokenizer.batch_encode_plus(texts,max_length=self.max_src_length,padding='longest',truncation=True,return_tensors='pt')
		return o['input_ids'].to(self.device,dtype=torch.long),o['attention_mask'].to(self.device,dtype=torch.long)

	def _to_tensor(self,batch):
		src_ids,src_msks=self.text_to_tensor([p[0] for p in batch])
		tgt = self.tokenizer.batch_encode_plus([p[1] for p in batch], max_length=self.max_src_length, padding='longest',truncation=True,return_tensors='pt')
		y=tgt['input_ids']
		y_ids = y[:,:-1]
		lm_labels = y[:,1:].clone().detach()
		lm_labels[y[:,1:]==self.tokenizer.pad_token_id] = -100
		tgt_ids = y_ids.to(self.device,dtype=torch.long)
		tgt_msks = tgt['attention_mask'][:,:-1].to(self.device,dtype=torch.long)
		labels = lm_labels.to(self.device,dtype=torch.long)

		example_size=len(batch[0][2])
		choice=random.choices(range(example_size),k=self.contrast_number)
		texts=[]
		for i in range(self.contrast_number):
			texts.extend([p[2][choice[i]] for p in batch])


		con_ids,con_msks=self.text_to_tensor(texts)
		return (src_ids,tgt_ids,con_ids,src_msks,tgt_msks,con_msks,labels)

	def __next__(self):
		if self.res and self.index==self.n_batches:
			batches=self.dataset[self.index*self.batch_size:len(self.dataset)]
			self.index+=1
			batches=self._to_tensor(batches)
			return batches

		elif self.index>self.n_batches:
			self.index=0
			raise StopIteration
		else:
			batches=self.dataset[self.index*self.batch_size:(self.index+1)*self.batch_size]
			self.index+=1
			batches=self._to_tensor(batches)
			return batches
	
	def shuffle_data(self):
		random.shuffle(self.dataset)
		self.index=0

	def __iter__(self):
		return self

	def __len__(self):
		if self.res:
			return self.n_batches+1
		else:
			return self.n_batches

