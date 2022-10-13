import os
import codecs
from pyrouge import Rouge155
from rouge import Rouge

#home_path=os.path.expanduser('~')
#r = Rouge155('%s/ROUGE-1.5.5/' % home_path)
#use pyrouge package to compute rouge. https://github.com/bheinzerling/pyrouge
#pyrouge is based on the official code ROUGE-1.5.5.pl
def Rouge_pyrouge(hyps,refs,path='result'): 
	if not os.path.exists(path):
		os.makedirs(path)

	# write ref and hyp
	for idx,(ref,hyp) in enumerate(zip(refs,hyps)):
		with codecs.open(os.path.join(path, 'ref.' + str(idx) + '.txt'), 'w', encoding="UTF-8") as f:
			f.write(Rouge155.convert_text_to_rouge_format(ref))
		with codecs.open(os.path.join(path, 'hyp.' + str(idx) + '.txt'), 'w', encoding="UTF-8") as f:
			f.write(Rouge155.convert_text_to_rouge_format(hyp))

	r.system_dir = path
	r.model_dir = path
	r.system_filename_pattern = 'hyp.(\d+).txt'
	r.model_filename_pattern = 'ref.#ID#.txt'

	output = r.evaluate()
	#print(output)
	output_dict = r.output_to_dict(output)
	return output_dict


#py-rouge package. This rouge-score is slightly different from the above one. The author pointed out that ROUGE1.5.5 has a bug and fixed it.
#https://github.com/Diego999/py-rouge
def Rouge_py_rouge(hyps,refs):
	evaluator=Rouge(metrics=['rouge-n','rouge-l'],max_n=2)
	score=evaluator.get_scores(hyps,refs)
	return score
