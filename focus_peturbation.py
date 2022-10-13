from collections import defaultdict,Counter
import argparse
import json
from flair.models import SequenceTagger
from flair.data import Sentence
import random
import rouge
import os
from tqdm import tqdm

KEY_PARSE=['VP','NN','NP']
stopwords= ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']

def keyphrase_extract(source,target):
	d=Counter()
	for p,t in source:
		d[t]+=1
	focus_labels=[]
	focus_exists=False
	for i,(p,t) in enumerate(target):
		if t in d:
			if p in KEY_PARSE:
				focus_labels.append(i)
				focus_exists=True
	return focus_labels,focus_exists



def com_generate_prob(l):
	g=[0]*l
	pone=1/l
	count=0
	for i in range(l):
		p=random.random()
		if p<pone:
			g[i]=1
			count+=1
	if count==0:
		r=random.choice(range(l))
		g[r]=1
	
	return g

def half_search(v,p,f,t):
	if(f==t):
		return v[f+1][0]
	c=(f+t)//2
	if v[c][1]>p:
		return half_search(v,p,f,c-1)
	elif v[c][1]==p:
		return v[c+1][0]
	else:
		if v[c+1][1]>p:
			return v[c+1][0]
		else:
			return half_search(v,p,c+1,t)

def peturbation(target,focus_labels,vocabs,n):
	generated_examples=[]
	for i in range(n):
		example=[t[1] for t in target]
		is_generate=com_generate_prob(len(focus_labels))
		for idx,j in enumerate(focus_labels):
			if is_generate[idx]:
				fp=target[j][0]
				while True:
					prob=random.random()
					random_token=half_search(vocabs[fp],prob,0,len(vocabs[fp])-1)
					if random_token!=example[j]:
						example[j]=random_token
						break
		text=' '.join(example)
		generated_examples.append(text)
	return generated_examples

def get_chunk_parse(args,sent, use_stop_class=True):
	parse = [tok.labels[0].value for tok in sent.tokens]
	toks = [tok.text for tok in sent.tokens]


	parse_filtered = []
	toks_filtered = []

	curr_tag = None
	span_text = []
	for i in range(len(parse)):
		tok = toks[i]

		if tok.lower() not in stopwords:
			tag_parts = parse[i].split('-')

			if args.tagger=='pos':
				tag_parts = [''] + tag_parts

			if len(tag_parts) > 1:
				this_tag = tag_parts[1]
				if this_tag == curr_tag:
					span_text.append(tok)
				elif curr_tag is not None:
					parse_filtered.append(curr_tag)
					toks_filtered.append(" ".join(span_text))
					curr_tag = this_tag
					span_text = [tok]
				else:
					curr_tag = this_tag
					span_text = [tok]

			else:
				if len(span_text) > 0:
					parse_filtered.append(curr_tag)
					toks_filtered.append(" ".join(span_text))
				curr_tag = None
				span_text = []
				parse_filtered.append(curr_tag)
				toks_filtered.append(tok)

		else:
			if len(span_text) > 0:
				parse_filtered.append(curr_tag)
				toks_filtered.append(" ".join(span_text))
			curr_tag = None
			parse_filtered.append('STOP' if use_stop_class else None)
			toks_filtered.append(tok)
			span_text = []


	if len(span_text) > 0:
		parse_filtered.append(curr_tag)
		toks_filtered.append(" ".join(span_text))

	parse_filtered = [tag if tag is not None else toks_filtered[i] for i, tag in enumerate(parse_filtered)]
	return parse_filtered, toks_filtered


def to_sentence(s,max_length):
	s=Sentence(s)
	if len(s)>args.max_length:
		s=Sentence([stext.text for stext in s[0:args.max_length]])
	return s


def read_json(dataset,max_length):
	data=[]
	rg=rouge.Rouge()
	with open(dataset,'r') as f:
		for l in f.readlines():
			j=json.loads(l)
			rl=rg.get_scores(j['chq'],j['faq'])[0]['rouge-l']['f']
			if rl>0.8:
				continue
			data.append((j['chq'],j['faq'],to_sentence(j['chq'],max_length),to_sentence(j['faq'],max_length)))
		f.close()
	return data
	

def run(args):
	data=read_json(args.dataset,args.max_length)
	vocabs=defaultdict(Counter)
	source_all=[]
	target_all=[]
	tagger=SequenceTagger.load(args.tagger)

	print('construct vocabulary...')
	batch_num=(len(data)-1)//args.batch +1
	for b_idx in tqdm(range(batch_num)):
		batch=data[b_idx*args.batch:(b_idx+1)*args.batch]
		source_sentence=[p[2] for p in batch]
		tgt_sentence=[p[3] for p in batch]

		tagger.predict(source_sentence)
		tagger.predict(tgt_sentence)
		for s,t in zip(source_sentence,tgt_sentence):
			source_parse,source_toks=get_chunk_parse(args,s)
			target_parse,target_toks=get_chunk_parse(args,t)
			source_concat=[(sp,st) for sp,st in zip(source_parse,source_toks)]
			target_concat=[(tp,tt) for tp,tt in zip(target_parse,target_toks)]
			for tokens in target_concat:
				vocabs[tokens[0]][tokens[1]]+=1
			source_all.append(source_concat)
			target_all.append(target_concat)
	#calcuate vocab probility
	vocabs = {tag: [(w,count) for w,count in vocab.items()] for tag,vocab in vocabs.items()}
	vocabs = {tag: sorted(vocab, reverse=True, key=lambda x: x[1])[:5000] for tag,vocab in vocabs.items()}
	vocabs_size = {tag: sum([x[1] for x in vocab]) for tag,vocab in vocabs.items()}
	vocabs = {tag: [(x[0],x[1]/vocabs_size[tag]) for x in vocab] for tag,vocab in vocabs.items()}
	vocabs_new={}
	for tag,vocab in vocabs.items():
		vn=[('',0)]
		last=0
		for w,p in vocab:
			last=last+p
			vn.append((w,last))
		vocabs_new[tag]=vn
	vocabs=vocabs_new

	w1=open(args.output+'.peturb','w')
	w2=open(args.output+'.nopeturb','w')

	outjs_peturb=[]
	outjs_nopeturb=[]
	print('generating peturbations...')
	for (src,tgt,_,__),source,target in tqdm(zip(data,source_all,target_all)):
		focus_labels,focus_exists=keyphrase_extract(source,target)
		if focus_exists:
			peturb_texts=peturbation(target,focus_labels,vocabs,args.sample_size)
			focus_texts=[target[i][1] for i in focus_labels]
			outj=json.dumps({'chq':src,'faq':tgt,'focus':focus_texts,'peturbations':peturb_texts})
			w1.write(outj+'\n')
		else:
			outj=json.dumps({'chq':src,'faq':tgt})
			w2.write(outj+'\n')

	w1.close()
	w2.close()

if __name__=='__main__':
	parser = argparse.ArgumentParser(description="generate hard negative examples for question summary")
	parser.add_argument("--sample_size", type=int, default=128, help="sample size")
	parser.add_argument("--max_length", type=int, default=256, help="max_length")
	parser.add_argument("--batch", type=int, default=64, help="batch")
	parser.add_argument("--dataset",type=str,default="",help="dataset path")
	parser.add_argument("--output",type=str,default="",help="output path")
	parser.add_argument("--tagger",type=str,default="chunk",help="tagger")
	args=parser.parse_args()
	run(args)


