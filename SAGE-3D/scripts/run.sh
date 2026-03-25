#!/bin/bash
#SBATCH --mail-user=snehapaulputul@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes 1
#SBATCH --gpus-per-node=4  
#SBATCH --cpus-per-task=12 # change this parameter to 2,4,6,... and increase "--num_workers" accordingly to see the effect on performance
#SBATCH --time=24:00:00   # %%%%%%%%%%%%%%%%% MUST CHANGE %%%%%%%%%%%%%%%
#SBATCH --output=outputs/stage_1_%j.out  # %%%%%%%%%%%%%%%%% MUST CHANGE %%%%%%%%%%%%%%%
#SBATCH --account=def-zpatters

cd ..
module load StdEnv/2023 cuda/12.6 cmake gcc arrow opencv rust 
source /home/sneha28/scratch/ENEL/bin/activate
# source /home/sneha28/scratch/3d/bin/activate


export TRANSFORMERS_OFFLINE=1 
export HF_DATASETS_OFFLINE=1 
export HF_HUB_OFFLINE=1 
export WANDB_MODE=offline


echo "Activating virtual environment"
master_port=$((RANDOM % (65535 - 49152 + 1) + 49152))
# Get the filename without extension
export WANDB_MODE=disabled
filename=codebook_no_mask_recheck

dir_path=.
model_name_or_path=./checkpoints/PointLLM_7B_v1.1_init
data_path=./data/objaverse_data
anno_path=./data/anno_data/PointLLM_brief_description_660K_filtered.json # or PointLLM_brief_description_660K.json (including val sets)
output_dir=./outputs/Train_stage1/$filename

#srun -p optimal --quotatype=spot --gres=gpu:4 -J leo
#CUDA_VISIBLE_DEVICES=1,2,3,7
#CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4 --master_port=$master_port pointllm/train/train_mem.py \
# cmd="PYTHONPATH=$dir_path:$PYTHONPATH \
# CUDA_VISIBLE_DEVICES=3 python pointllm/train/train_mem.py \

# export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4 --master_port=$master_port pointllm/train/train_mem.py \
    --model_name_or_path $model_name_or_path \
    --data_path $data_path \
    --anno_path $anno_path \
    --output_dir $output_dir \
    --version v1 \
    --model_max_length 2048 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 4e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --bf16 True \
    --fix_llm True \
    --fix_pointnet False \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --run_name $filename \
    --mm_projector_lr 4e-4 \
    --vision_tower_lr 4e-4 \
    --group_size 81 \
    --num_stages 1 \
    --embed_dim 288 \
    --LGA_dim 2 2 2 \
    --input_points 1024 \
    --tune_layer 4 \
    --use_color True \
    --recon_fp 0 \
    --mae_fp 1 \
    --mask_dim 4096 \
    --mask_ratio 0.0 \
    --mae_feature 0 \
    --recon_feature 0 \
    --pos_embed_mae 0 \
    --pos_embed_dim 4096 \
    --recon_pos 1 \
    --alpha 1.0 \
    --beta 0.0 \
    --gamma 0.0 \
    --vq_cost 0.5 \
    --codebook_size 8192 \
    --commitment_cost 0.25 \

echo "End of  training..."

### Stage 2 training
export NCCL_TIMEOUT=3600

master_port=$((RANDOM % (65535 - 49152 + 1) + 49152))
# Get the filename without extension
export WANDB_MODE=disabled

filename=codebook_no_mask_recheck

dir_path=.
model_name_or_path=./outputs/Train_stage1/codebook_no_mask_recheck/checkpoint-15400 # Path to the output dir of stage 1 training
data_path=./data/objaverse_data
anno_path=./data/anno_data/PointLLM_complex_instruction_70K.json
output_dir=./outputs/Train_stage2/$filename
# CUDA_VISIBLE_DEVICES=0 python pointllm/train/train_mem.py \

# export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4 --master_port=$master_port pointllm/train/train_mem.py \
    --model_name_or_path $model_name_or_path \
    --data_path $data_path \
    --anno_path $anno_path \
    --output_dir $output_dir \
    --version v1 \
    --model_max_length 2048 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --eval_steps 100 \
    --save_strategy "steps" \
    --save_steps 10000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --bf16 True \
    --fix_llm False \
    --fix_pointnet False \
    --report_to tensorboard \
    --run_name $filename \
    --gradient_checkpointing True \
    --stage_2 True \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --conversation_types "detailed_description" "single_round" "multi_round" \
    --mm_projector_lr 2e-5 \
    --vision_tower_lr 2e-5 \
    --group_size 81 \
    --num_stages 1 \
    --embed_dim 288 \
    --LGA_dim 2 2 2 \
    --input_points 1024 \
    --tune_layer 0 \
    --mask_ratio 0.0 \
    --recon_fp 0 \
    --mae_fp 1 \
    --mask_dim 4096 \
    --mae_feature 0 \
    --recon_feature 0 \
    --recon_pos 1 \
    --pos_embed_mae 0 \
    --pos_embed_dim 4096 \
    --use_color True \
    --vq_cost 0.5 \
    --codebook_size 8192 \
    --commitment_cost 0.25 \


### Evaluation
# Model and log paths
MODEL_NAME="./outputs/Train_stage2/codebook_no_mask_recheck"
LOG_SUFFIX="eval"
LOG_DIR="./outputs/new_eval_logs"
LOG_EDIR="./outputs/new_eval_logs"

# export PYTHONPATH=.

# Object captioning on Objaverse
CUDA_VISIBLE_DEVICES=0 python pointllm/eval/eval_objaverse.py --model_name $MODEL_NAME --task_type captioning --prompt_index 2 &

# Open Vocabulary Classification on Objaverse
CUDA_VISIBLE_DEVICES=1 python pointllm/eval/eval_objaverse.py  --model_name $MODEL_NAME --task_type classification --prompt_index 0 


### Traditional Evaluation
deactivate ENEL
source /home/sneha28/scratch/3d/bin/activate

CUDA_VISIBLE_DEVICES=0 python pointllm/eval/traditional_evaluator.py --results_path ./outputs/Train_stage2/codebook_no_mask_recheck/evaluation/PointLLM_brief_description_val_200_GT_Objaverse_captioning_prompt2.json