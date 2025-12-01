GPU_NUM=8
CFG='./config/config_cfg.py'
OUTPUT_DIR='./test_output'
PRETRAIN_MODEL_PATH= #Your pretrained model path


NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

python -m torch.distributed.launch  --nproc_per_node=${GPU_NUM} eval.py \
        --output_dir ${OUTPUT_DIR} \
        --eval \
        -c ${CFG} \
        --top_k 5 \
        --batch 8 \
        --pretrain_model_path ${PRETRAIN_MODEL_PATH} \


