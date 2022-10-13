import argparse
from qsmoco import QsMoco
from transformers import AutoTokenizer
from train_eval import train,test
import load_data_for_contrast
import load_data
import os
import torch
if __name__ == "__main__":
	parser = argparse.ArgumentParser(
        description="QFCL")
	parser.add_argument("--dataset", type=str, default="meqsum_contrast")
	parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
	parser.add_argument("--learning_rate", type=float, default=0.00001,
                        help="Learnign rate")
	parser.add_argument("--batch_size", type=int, default=4, metavar="N",
                        help="input batch size for training (default: 4)")
	parser.add_argument("--log_interval", type=int, default=100, metavar="N",
                        help="how many batches to wait before logging training status")
	parser.add_argument("--cuda", type=int,default=0,
                        help="gpu index,-1 means cpu")
	parser.add_argument("--output_dir", type=str,default="../train/meqsum_contrast_focus",
                        help="path to save the output to")
	parser.add_argument("--need_improve", type=int, default=20)
	parser.add_argument("--model_path", type=str, default=None)
	parser.add_argument("--mod", type=str, default="train")
	parser.add_argument("--model", type=str, default="facebook/bart-large",help="choose from [facebook/bart-large,microsoft/prophetnet-large-uncased]")
	parser.add_argument("--num_beams", type=float, default=4)
	parser.add_argument("--max_src_length", type=float, default=128)
	parser.add_argument("--max_tgt_length", type=float, default=20)
	parser.add_argument("--alpha", type=float, default=1)
	parser.add_argument("--beta", type=float, default=0.5)
	parser.add_argument("--contrast_K", type=int, default=4096)
	parser.add_argument("--contrast_T", type=float, default=0.07)
	parser.add_argument("--contrast_M", type=float, default=0.999)
	parser.add_argument("--contrast_number", type=int, default=64)
	parser.add_argument("--freeze_encoder", action="store_true")
	parser.add_argument("--contrast_decoder", action="store_true")
	parser.add_argument("--do_train_contrast", action="store_true")
	parser.add_argument("--no_hard", action="store_true")
	parser.add_argument("--save_each", action="store_true")

	args = parser.parse_args()
	device='cpu'
	if args.cuda!=-1:
		device='cuda:{}'.format(args.cuda)
	args.device=device

	tokenizer=AutoTokenizer.from_pretrained(args.model)
	hard_set=load_data_for_contrast.load_dataset(os.path.join('../dataset/jsons/',args.dataset),args)
	hard_set=load_data_for_contrast.DatasetIterater(tokenizer,hard_set,args.batch_size,device,args.contrast_number,args.max_src_length)
	
	train_set,val_set,test_set=load_data.load_dataset(os.path.join('../dataset/jsons/',args.dataset),args)
	train_set=load_data.DatasetIterater(tokenizer,train_set,args.batch_size,device,args.max_src_length)
	val_set = load_data.DatasetIterater(tokenizer,val_set, args.batch_size, device,args.max_src_length)
	test_set = load_data.DatasetIterater(tokenizer,test_set, args.batch_size, device,args.max_src_length)

	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	
	model=QsMoco(args)
	model=model.to(device)
	if args.model_path is not None:	
		model.load_state_dict(torch.load(args.model_path,map_location=device))

	if args.mod=='train':
		train(args,model,tokenizer,hard_set,train_set,val_set,test_set)
	elif args.mod=='test':
		test(args,model,tokenizer,test_set)
		
