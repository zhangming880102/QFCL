import json
import os
import torch
import random
import copy
def load_dataset(dataset_dir,args):

	def load(path):
		contents=[]
		with open(path,'r') as f:
			for line in f.readlines():
				j=json.loads(line)
				chq=j['chq'].strip()
				faq=j['faq'].strip()
				contents.append((chq,faq))
		return contents

	train_set=load(os.path.join(dataset_dir,'contrast.noperturb'))
	val_set=load(os.path.join(dataset_dir,'val.json'))
	test_set=load(os.path.join(dataset_dir,'test.json'))

	return train_set,val_set,test_set


class DatasetIterater(object):
	def __init__(self,tokenizer,dataset,batch_size,device,max_src_length):
		self.batch_size=batch_size
		self.dataset=dataset
		self.device=device
		self.tokenizer=tokenizer
		self.n_batches=(len(dataset)-1) //batch_size
		self.res=False
		self.max_src_length=max_src_length
		if len(dataset)% self.n_batches!=0:
			self.res=True
		self.index=0

	def _to_tensor(self,batch):
		src=self.tokenizer.batch_encode_plus([p[0] for p in batch],max_length=self.max_src_length,padding='longest',truncation=True,return_tensors='pt')
		tgt = self.tokenizer.batch_encode_plus([p[1] for p in batch], max_length=self.max_src_length, padding='longest',truncation=True,return_tensors='pt')
		y=tgt['input_ids']
		y_ids = y[:,:-1]
		lm_labels = y[:,1:].clone().detach()
		lm_labels[y[:,1:]==self.tokenizer.pad_token_id] = -100
		src_id=src['input_ids'].to(self.device,dtype=torch.long)
		tgt_id = y_ids.to(self.device,dtype=torch.long)
		src_mask = src['attention_mask'].to(self.device,dtype=torch.long)
		tgt_mask = tgt['attention_mask'][:,:-1].to(self.device,dtype=torch.long)
		labels = lm_labels.to(self.device,dtype=torch.long)
		return (src_id,tgt_id,src_mask,tgt_mask,labels)

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

