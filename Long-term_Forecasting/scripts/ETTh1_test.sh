
export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export GPT2_MODEL_PATH=${GPT2_MODEL_PATH:-/root/.cache/huggingface/hub/models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e}
export GPT2_LOCAL_FILES_ONLY=${GPT2_LOCAL_FILES_ONLY:-1}

seq_len=336
gpt_layers=6
multi_patch=16,24,48
model=MultiPeriodGPT4TS
run_time=$(date +%Y%m%d%H)
mkdir -p ./checkpoints
: > ./checkpoints/result_${run_time}.txt

for percent in 100
do
for pred_len in 96 192 336 720
do
for lr in 0.0001
do

python main.py \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_${model}_${run_time} \
    --data ett_h \
    --seq_len $seq_len \
    --label_len 168 \
    --pred_len $pred_len \
    --batch_size 256 \
    --lradj type4 \
    --learning_rate $lr \
    --train_epochs 10 \
    --decay_fac 0.5 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 7 \
    --c_out 7 \
    --freq 0 \
    --patch_size 16 \
    --multi_patch $multi_patch \
    --stride 8 \
    --percent $percent \
    --gpt_layers $gpt_layers \
    --itr 3 \
    --model $model \
    --tmax 20 \
    --cos 1 \
    --is_gpt 1 \
    --run_time $run_time

done
done
done
