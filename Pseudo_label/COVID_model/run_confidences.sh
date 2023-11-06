#!/bin/bash
python model_confidences.py --modality BCV \
	--num_units 64 \
	--lr1 1e-5 \
	--lr2 5e-5 \
        --data_name data/unlabelled_all_ \
	--is_diff True \
	--train_vgg True \
	--trained_layers 12 \
	--train_name covid_model
