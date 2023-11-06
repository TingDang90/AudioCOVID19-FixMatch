#!/bin/bash
python get_predictions.py --modality BCV \
	--num_units 64 \
	--lr1 1e-5 \
	--lr2 5e-5 \
	--loss_weight 10 \
	--data_name data/audio_0426En \
        --unlabelled_data_name data/unlabelled_all_ \
	--is_diff True \
	--train_vgg True \
	--trained_layers 12 \
	--train_name covid_model_fix \
        --pseudo_label_threshold 0.9
