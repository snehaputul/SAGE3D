# Point Cloud as a Foreign Language for Multi-modal Large Language Model

Official repository for the paper "Point Cloud as a Foreign Language for Multi-modal Large Language Model"
## 🏠 Abstract
Multi-modal large language models (MLLMs) have shown remarkable progress in integrating visual and linguistic understanding. Recent efforts have extended these capabilities to 3D understanding through encoder-based architectures that rely on pre-trained 3D encoders to extract geometric features. However, such approaches suffer from semantic misalignment between geometric and linguistic spaces, resolution sensitivity, and substantial computational overhead.
In this work, we present SAGE, the first end-to-end 3D MLLM that directly processes raw point clouds without relying on a pre-trained 3D encoder. Our approach introduces a lightweight 3D tokenizer that combines geometric sampling and neighbourhood aggregation with vector quantization to convert point clouds into discrete tokens—treating 3D data as a foreign language that naturally extends the LLM’s vocabulary.
Furthermore, to enhance the model’s reasoning capability on complex 3D tasks, we propose a preference optimization training strategy with a semantic alignment–based reward, specifically designed for open-ended 3D question answering where responses are descriptive. Extensive experiments across diverse 3D understanding benchmarks demonstrate that our end-to-end approach outperforms existing encoder-based methods while offering significant advantages in computational efficiency, generalization across LLM backbones, and robustness to input resolution variations.</b>



## 📦 Training and Evaluation
### Installation
To start: 
1. Clone this repository.
```bash
cd SAGE
```
2. Install packages
```bash
conda create -n SAGE python=3.10 -y
conda activate SAGE
pip install --upgrade pip  # enable PEP 660 support
pip install -e .

# * for training
pip install ninja
pip install flash-attn

# * for chamfer_dist
git clone https://github.com/Pang-Yatian/Point-MAE.git
cd ./extensions/chamfer_dist
python setup.py install --user
```

### Data Preparation
#### Objaverse Training Data
1. Download the two compressed files of 660K Objaverse colored point clouds [here](https://huggingface.co/datasets/RunsenXu/PointLLM/tree/main). They require about 77GB of storage space.
2. Run the following command to merge the two files into one and uncompress it. This will produce a folder named `8192_npy` containing 660K point cloud files named `{Objaverse_ID}_8192.npy`. Each file is a numpy array with dimensions (8192, 6), where the first three dimensions are `xyz` and the last three dimensions are `rgb` in [0, 1] range.
```bash
cat Objaverse_660K_8192_npy_split_a* > Objaverse_660K_8192_npy.tar.gz
tar -xvf Objaverse_660K_8192_npy.tar.gz
```
3. In `SAGE` folder, create a folder `data` and create a soft link to the uncompressed file in the directory.
```bash
cd SAGE
mkdir data
ln -s /path/to/8192_npy data/objaverse_data
```

#### Instruction-Following Data
1. In `SAGE/data` folder, create a directory named `anno_data`.
2. Our instruction-following data, including both the simple-description and complex instructions, can be downloaded [here](https://huggingface.co/datasets/RunsenXu/PointLLM). If you have difficulty downloading the data (e.g. network issue), please email the authors.
- The simple-description data has 660K samples and the complex instructions have 70K samples.
- Both training data are based on the Objaverse dataset.
- The complex instructions are generated with GPT-4.
3. Put the data files in the `anno_data` directory. The directory should look like this:
```bash
SAGE/data/anno_data
├── PointLLM_brief_description_660K_filtered.json
├── PointLLM_brief_description_660K.json
└── PointLLM_complex_instruction_70K.json
```
4. Note, the `PointLLM_brief_description_660K_filtered.json` is filtered from `PointLLM_brief_description_660K.json` by removing the 3000 objects we reserved as the validation set. 

#### Evaluation Data
1. Download the referencing GT `PointLLM_brief_description_val_200_GT.json` we use for the benchmarks on Objaverse dataset [here](https://huggingface.co/datasets/RunsenXu/PointLLM/blob/main/PointLLM_brief_description_val_200_GT.json), and put it in `SAGE/data/anno_data`.

### Training
#### Download the Initial LLM Weight
1. In `SAGE` folder, create a directory named `checkpoints`.
2. Download the pre-trained LLM: [
PointLLM_7B_v1.1_init](https://huggingface.co/RunsenXu/PointLLM_7B_v1.1_init/tree/main). Put them in the `checkpoints` directory.

#### Start Training
1. For stage-1 training, simply run:
```bash
dir_path=.
model_name_or_path=./checkpoints/PointLLM_7B_v1.1_init
data_path=./data/objaverse_data
anno_path=./data/anno_data/PointLLM_brief_description_660K_filtered.json # or PointLLM_brief_description_660K.json (including val sets)
output_dir=./outputs/Train_stage1/$filename

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
```

2. After stage-1 training, start stage-2 training:
```bash
dir_path=.
model_name_or_path=./outputs/Train_stage1/$filename/checkpoint-15400  # Path to the output dir of stage 1 training
data_path=./data/objaverse_data
anno_path=./data/anno_data/PointLLM_complex_instruction_70K.json
output_dir=./outputs/Train_stage2/$filename

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
```
### Evaluation
```bash
# Model and log paths
MODEL_NAME="./outputs/Train_stage2/$filename"
LOG_SUFFIX="eval"
LOG_DIR="./outputs/new_eval_logs"
LOG_EDIR="./outputs/new_eval_logs"

# Object captioning on Objaverse
CUDA_VISIBLE_DEVICES=0 python pointllm/eval/eval_objaverse.py --model_name $MODEL_NAME --task_type captioning --prompt_index 2 &

# Open Vocabulary Classification on Objaverse
CUDA_VISIBLE_DEVICES=1 python pointllm/eval/eval_objaverse.py  --model_name $MODEL_NAME --task_type classification --prompt_index 0 


### Traditional Evaluation
CUDA_VISIBLE_DEVICES=0 python pointllm/eval/traditional_evaluator.py --results_path ./outputs/Train_stage2/$filename/evaluation/PointLLM_brief_description_val_200_GT_Objaverse_captioning_prompt2.json

# Object captioning on Objaverse
CUDA_VISIBLE_DEVICES=1 python pointllm/eval/eval_objaverse.py --model_name $MODEL_NAME --task_type captioning --prompt_index 2 > $LOG_EDIR/try_obj_${LOG_SUFFIX}.log 2>&1 &

# Open Vocabulary Classification on Objaverse
CUDA_VISIBLE_DEVICES=2 python pointllm/eval/eval_objaverse.py  --model_name $MODEL_NAME --task_type classification --prompt_index 0 > $LOG_EDIR/try_objcls_${LOG_SUFFIX}.log 2>&1 &
```
3. After inferencing, the results will be saved in `{model_name}/evaluation` as a dict with the following format:
```bash
{
  "prompt": "",
  "results": [
    {
      "object_id": "",
      "ground_truth": "", 
      "model_output": "",
      "label_name": "" # only for classification on modelnet40
    }
  ]
}
```


## 📄 License
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a>
This work is under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

## 👏 Acknowledgements
- [Point-Bind & Point-LLM](https://arxiv.org/abs/2309.00615)
- [PointLLM](https://arxiv.org/abs/2308.16911)
- [ShapeLLM](https://arxiv.org/abs/2402.17766)
- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [Vicuna](https://github.com/lm-sys/FastChat)
- [Objaverse](https://objaverse.allenai.org)
- [Cap3D](https://github.com/crockwell/Cap3D/)
