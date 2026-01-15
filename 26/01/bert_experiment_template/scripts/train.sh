dataset_name=$1
raw_model_name=$2
batch_size=${3:-32}
desc=${4:""}
epochs=1

output_home=output

python ../py_scripts/train.py \
  --do_train \
  --output_dir=${output_home}/$dataset_name/direct/$raw_model_name/batch${batch_size}${desc} \
  --dataset_name=$dataset_name \
  --model_name_or_path=$raw_model_name \
  --per_device_train_batch_size=$batch_size \
  --per_device_eval_batch_size=$batch_size \
  --num_train_epochs=$epochs \
  --eval_strategy=epoch \
  --save_strategy=epoch \
  --warmup_ratio=0.1 \
  --logging_steps=50 \
  --lr_scheduler_type=cosine \
  --load_best_model_at_end=True \
  --metric_for_best_model=f1 \
  --save_total_limit=2 \
  --report_to=tensorboard \
  --remove_unused_columns=False \
  --fp16=True \
  --fp16_full_eval=True
