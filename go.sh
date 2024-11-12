#!/bin/bash
# Run script
method="${1:-tmp}"
size=256
epochs=150
val_last=50

root_path=${HOME}/codes/cosod

# Train
CUDA_VISIBLE_DEVICES=${2:-0} python train.py --trainset DUTS_class+coco-seg --size ${size} --ckpt_dir ckpt/${method} --epochs ${epochs}


# Test & Eval
# If dirs of performance and predictions already exist, remove them.
if [ "${method}" ] ;then
    rm -rf evaluation/${method}
    rm -rf ${root_path}/preds/${method}
fi

step=10
for ((ep=${epochs};ep>${epochs}-${val_last};ep-=${step}))
do
pred_dir=${root_path}/preds/${method}/ep${ep}
# [ ${ep} -gt $[${epochs}-${val_last}] ] && CUDA_VISIBLE_DEVICES=$2 python evaluation/main.py --model_dir ${method}/ep$[${ep}-${step}] --txt_name ${method}; \
CUDA_VISIBLE_DEVICES=${2:-0} python test.py --pred_dir ${pred_dir} --ckpt ckpt/${method}/ep${ep}.pth --size ${size}
done

# python evaluation/main.py --model_dir ${method}/ep$[${ep}-${step}] --txt_name ${method}
python evaluation/main.py --model_dir ${method} --txt_name ${method}

nvidia-smi
hostname
