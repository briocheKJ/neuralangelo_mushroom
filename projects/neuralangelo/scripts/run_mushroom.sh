EXPERIMENT=mushroom
GROUP=mushroom_test
CONFIG=projects/neuralangelo/configs/${EXPERIMENT}.yaml
GPUS=2  # use >1 for multi-GPU training!

NAME=vr_room
CUDA_VISIBLE_DEVICES=0,4 torchrun --rdzv_endpoint=localhost:29405 --nproc_per_node=${GPUS} train.py \
    --logdir=/NASdata/lkj/log/neuralangelo/${GROUP}/${NAME} \
    --config=${CONFIG} \
    --show_pbar --wandb --wandb_name iclr_mushroom
    --data.root=/NASdata/lkj/dataset/mushroom/${NAME}/iphone/long_capture \

NAME=coffee_room
CUDA_VISIBLE_DEVICES=1,5 torchrun --rdzv_endpoint=localhost:29406 --nproc_per_node=${GPUS} train.py \
    --logdir=/NASdata/lkj/log/neuralangelo/${GROUP}/${NAME} \
    --config=${CONFIG} \
    --show_pbar --wandb --wandb_name iclr_mushroom \
    --data.root=/NASdata/lkj/dataset/mushroom/${NAME}/iphone/long_capture

NAME=kokko
CUDA_VISIBLE_DEVICES=2,6 torchrun --rdzv_endpoint=localhost:29407 --nproc_per_node=${GPUS} train.py \
    --logdir=/NASdata/lkj/log/neuralangelo/${GROUP}/${NAME} \
    --config=${CONFIG} \
    --show_pbar --wandb --wandb_name iclr_mushroom \
    --data.root=/NASdata/lkj/dataset/mushroom/${NAME}/iphone/long_capture

NAME=honka
CUDA_VISIBLE_DEVICES=3,7 torchrun --rdzv_endpoint=localhost:29408 --nproc_per_node=${GPUS} train.py \
    --logdir=/NASdata/lkj/log/neuralangelo/${GROUP}/${NAME} \
    --config=${CONFIG} \
    --show_pbar --wandb --wandb_name iclr_mushroom \
    --data.root=/NASdata/lkj/dataset/mushroom/${NAME}/iphone/long_capture

NAME=sauna
CUDA_VISIBLE_DEVICES=0,4 torchrun --rdzv_endpoint=localhost:29409 --nproc_per_node=${GPUS} train.py \
    --logdir=/NASdata/lkj/log/neuralangelo/${GROUP}/${NAME} \
    --config=${CONFIG} \
    --show_pbar --wandb --wandb_name iclr_mushroom \
    --data.root=/NASdata/lkj/dataset/mushroom/${NAME}/iphone/long_capture