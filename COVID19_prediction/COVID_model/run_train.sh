#!/bin/bash
python model_train_vad.py --modality BCV \
	--num_units 64 \
	--lr1 1e-5 \
	--lr2 5e-5 \
	--loss_weight 10 \
	--data_name data/audio_0426En \
	--is_diff True \
	--train_vgg True \
	--trained_layers 12 \
	--train_name covid_model \
        --early_stop AUC \
        --train_data_portion 1.0 \
        --shuffle_vad y
