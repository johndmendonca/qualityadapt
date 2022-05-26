export TASK_NAME=Human
export MODEL_NAME=roberta-large

python QualityAdapt/train_fusion_metric.py \
	--model_name_or_path $MODEL_NAME \
	--task_name $TASK_NAME \
	--do_train \
	--do_eval \
	--do_predict \
	--max_seq_length 128 \
	--per_device_train_batch_size 8 \
	--learning_rate 1e-4 \
	--num_train_epochs 10.0 \
	--output_dir exp/$MODEL_NAME/$TASK_NAME \
	--overwrite_output_dir \
	--train_adapter \
	--adapter_config pfeiffer \
	--train_file data/dailydialog/dd_h_train.csv \
	--validation_file data/dailydialog/dd_h_dev.csv \
	--test_file data/dailydialog/dd_h_test.csv \
	--human_test_file data/dailydialog/ctx-response-human-scores-all-floor.csv \
	--u_adapter_path exp/roberta-largeUnderstandability/Understandability \
	--s_adapter_path exp/roberta-large/Sensibleness/Sensibleness