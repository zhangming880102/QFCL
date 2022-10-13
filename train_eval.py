from torch import nn
import torch
import argparse
import time
import os
import numpy as np
from rouge import Rouge
import random
from tqdm import tqdm
from collections import defaultdict
from rouge_util import Rouge_py_rouge

def train(args,model,tokenizer,hard_set,simple_set,val_set,test_set):
	start_time=time.time()
	model.train()
	optimizer =torch.optim.Adam(model.model.parameters(),lr=args.learning_rate)
	
	hard_len=len(hard_set)
	simple_len=len(simple_set)

	select_hard_prob=1
	select_order=[0]*hard_len+[1]*simple_len
	best_val_rl=0
	last_improve=0
	for e in range(args.epochs):
		hard_set.shuffle_data()
		simple_set.shuffle_data()
		random.shuffle(select_order)
		print('Epoch [{}/{}]'.format(e+1,args.epochs))
		sim_total=[0,0,0]
		sim_count=[0,0,0]
		total_loss=0
		for index,i in tqdm(enumerate(select_order)):
			if i==0:
				src_ids,tgt_ids,contrast_ids,src_mask,tgt_mask,contrast_mask,labels=hard_set.__next__()
				curriculum_p=random.random()
				if args.no_hard or curriculum_p >select_hard_prob:
					contrast_ids=None
					contrast_mask=None
			else:
				src_ids,tgt_ids,src_mask,tgt_mask,labels=simple_set.__next__()
				contrast_ids=None
				contrast_mask=None

			optimizer.zero_grad()
			loss,sims=model(src_ids,tgt_ids,src_mask,tgt_mask,labels=labels,device=args.device,train_contrast=args.do_train_contrast,hard_sample_ids=contrast_ids,hard_sample_mask=contrast_mask,alpha=args.alpha,beta=args.beta,contrast_decoder=args.contrast_decoder)
			loss.backward()
			optimizer.step()

			total_loss+=loss.item()
			if len(sims)>0:
				for idx,sim in enumerate(sims):
					sim_total[idx]+=sim.cpu().detach().item()
					sim_count[idx]+=1

		for idx in range(len(sim_total)):
			if sim_count[idx]!=0:
				sim_total[idx]/=sim_count[idx]
		simple_set.index=0	
		val_rl=evaluate(args,model,tokenizer,val_set)
		if val_rl>best_val_rl:
			best_val_rl=val_rl
			last_improve=e
			improve='*'
			model_path=os.path.join(args.output_dir,'epoch_'+str(e)+'.pt')
			torch.save(model.state_dict(),model_path)
			args.model_path=model_path
		else:
			improve=''
			if args.save_each:
				model_path=os.path.join(args.output_dir,'epoch_'+str(e)+'.pt')
				torch.save(model.state_dict(),model_path)

		use_time=time.time()-start_time
		print('epoch:{},Train loss:{:.4f},Sim:{},Val rl:{:.4f},time:{}{}'.format(e,total_loss/len(select_order),'|'.join(['{:.3f}'.format(sim) for sim in sim_total]),val_rl,use_time,improve))
		model.train()
				
		if e-last_improve> args.need_improve:
			break
	test(args,model,tokenizer,test_set)


def evaluate(args,model,tokenizer,val_set):
	model.eval()
	predict_all=[]
	target_all=[]
	with torch.no_grad():
		for src_ids,tgt_ids,src_mask,tgt_mask,labels in val_set:
			generate_ids=model.model.generate(input_ids=src_ids,attention_mask=src_mask,max_length=args.max_tgt_length,num_beams=args.num_beams,early_stopping=True)
			pred=[tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generate_ids]
			tgt = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in
					tgt_ids]

			predict_all.extend(pred)
			target_all.extend(tgt)

	score=Rouge_py_rouge(predict_all,target_all)
	return score['rouge-l']['f']

def test(args,model,tokenizer,test_set):
	model.load_state_dict(torch.load(args.model_path,map_location=args.device))
	model.eval()
	predict_all=[]
	target_all=[]
	with torch.no_grad():
		for src_ids,tgt_ids,src_mask,tgt_mask,labels in test_set:
			generate_ids=model.model.generate(input_ids=src_ids,attention_mask=src_mask,max_length=args.max_tgt_length,num_beams=args.num_beams,early_stopping=True)
			pred=[tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generate_ids]
			tgt = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in
					tgt_ids]
			predict_all.extend(pred)
			target_all.extend(tgt)

	score=Rouge_py_rouge(predict_all,target_all)
	print('R1:{:.4f},R2:{:.4f},Rl:{:.4f}'.format(score['rouge-1']['f'],score['rouge-2']['f'],score['rouge-l']['f']))
	
	with open(os.path.join(args.output_dir,'test.predict.txt'),'w') as w:
		for p in predict_all:
			w.write(p+'\n')
		w.close()
	
