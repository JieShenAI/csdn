# conda activate llm
cd LLaMA-Factory

# kw_arr=(llm_train_100 llm_train_500 llm_train_1000 llm_train_2000)
kw_arr=(llm_train_100 llm_train_500 llm_train_1000)


for kw in "${kw_arr[@]}"; do
    echo $kw
    CUDA_VISIBLE_DEVICES=0 API_PORT=8000 llamafactory-cli api \
        --model_name_or_path /home/jie/.cache/modelscope/hub/ZhipuAI/glm-4-9b-chat \
        --adapter_name_or_path ./saves/GLM-4-9B-Chat/lora/240731-${kw} \
        --template glm4 \
        --finetuning_type lora \
        --infer_backend vllm \
        --vllm_enforce_eager &

    python ../infer_eval.py ${kw} > ../logs/${kw}.log 2>&1
    # 杀掉服务进程
    pkill -f llamafactory
    echo "Stopped llamafactory"
done

# nohup bash eval.sh > eval.log 2>&1 &