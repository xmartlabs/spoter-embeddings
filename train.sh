#!/bin/sh
python -m train \
	--save_checkpoints_every -1 \
    --experiment_name "augment_rotate_75_x8" \
    --epochs 1300 \
	--optimizer "SGD" \
	--lr 0.001 \
    --batch_size 32 \
	--dataset_project "Sign Language Recognition" \
	--dataset_name "wlasl_mapped_mediapipe_only_landmarks_25fps" \
	--training_set_path "WLASL100_train_25fps.csv" \
	--validation_set_path "WLASL100_val_25fps.csv" \
	--vector_length 32 \
	--epoch_iters -1 \
	--scheduler_factor 0 \
	--hard_triplet_mining "in_batch" \
	--filter_easy_triplets \
	--triplet_loss_margin 1 \
	--dropout 0.2 \
	--start_mining_hard=200 \
	--hard_mining_pre_batch_multipler=16 \
	--hard_mining_pre_batch_mining_count=5 \
	--augmentations_prob=0.75 \
	--hard_mining_scheduler_triplets_threshold=0 \
	# --normalize_embeddings \
