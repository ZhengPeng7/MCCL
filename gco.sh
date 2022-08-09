#!/bin/bash
# Run script
method="$1"
size=256
epochs=100
val_last=40

# Train
CUDA_VISIBLE_DEVICES=$2 python train.py --trainset DUTS_class --size ${size} --ckpt_dir ckpt/${method} --epochs ${epochs}


# Test & Eval
if [ "$word" ] ;then
    rm -rf evaluation/${method}
    rm -rf /root/autodl-tmp/datasets/sod/preds/${method}
fi

step=5
for ((ep=${epochs};ep>${epochs}-${val_last};ep-=${step}))
do
pred_dir=/root/autodl-tmp/datasets/sod/preds/${method}/ep${ep}
# [ ${ep} -gt $[${epochs}-${val_last}] ] && CUDA_VISIBLE_DEVICES=$2 python evaluation/main.py --model_dir ${method}/ep$[${ep}-${step}] --txt_name ${method}; \
CUDA_VISIBLE_DEVICES=$2 python test.py --pred_dir ${pred_dir} --ckpt ckpt/${method}/ep${ep}.pth --size ${size}
done

# CUDA_VISIBLE_DEVICES=$2 python evaluation/main.py --model_dir ${method}/ep$[${ep}-${step}] --txt_name ${method}
CUDA_VISIBLE_DEVICES=$2 python evaluation/main.py --model_dir ${method} --txt_name ${method}

nvidia-smi
hostname
