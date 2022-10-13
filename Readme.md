Run the following command to get the hard negative samples:

python focus_peturbation.py --sample_size 128 --batch 64 --dataset ../dataset/jsons/meqsum/train.json --output ../dataset/jsons/meqsum_contrast/contrast

To train QFCL:
python run.py --do_train_contrast --output_dir ../models/meqsum --contrast_decoder --learning_rate 1e-5 --batch_size 16 --dataset meqsum_contrast --alpha 1 --beta 0.5
