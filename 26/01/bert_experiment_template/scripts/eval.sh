dataset_name=$1
raw_model_name=$2
batch_size=${3:-32}
local_model_dir=$4
#desc=${4:""}
epochs=3

output_home=output

python ../py_scripts/eval.py \
  --do_eval \
  --output_dir=${output_home}/$dataset_name/direct/$raw_model_name/batch${batch_size} \
  --dataset_name=$dataset_name \
  --model_name_or_path=$local_model_dir \
  --per_device_eval_batch_size=$batch_size \
  --remove_unused_columns=False \
  --fp16_full_eval=True
