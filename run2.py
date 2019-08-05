import os
import sys

def pruning_scratch_resnet20(gpu:int, regu): 
    command = 'CUDA_VISIBLE_DEVICES={0} python  pruning_cifar12.py  \
    ./data/cifar.python --dataset cifar10 --arch resnet20 \
    --save_path ./result/scrath/cifar10_res20_regu{1:.2f} \
    --epochs 200 \
    --schedule 60 120 160 \
    --gammas 0.2 0.2 0.2 \
    --learning_rate 0.1 --decay 0.0005 --batch_size 128 \
    --simi_ch ikk --mode regu --regu {1}\
    --layer_begin 0  --layer_end 54 --epoch_prune 1'.format(gpu, regu)
    # print(command)
    return command

def run_1(regu):
    command = pruning_scratch_resnet20(gpu=0, 
                                       regu=regu,
                                       )
    os.system(command)


if __name__ == "__main__":
    # run_1(0)
    # run_1(0)
    # run_1(0)
    # run_1(0.1)
    # run_1(0.1)
    # run_1(0.1)
    # run_1(0.2)
    run_1(0.2)
    # run_1(0.2)
    # run_1(0.3)
    # run_1(0.3)
    # run_1(0.3)
    # run_1(0.4)
    # run_1(0.4)
    # run_1(0.4)
    # run_1(0.5)
    # run_1(0.5)
    # run_1(0.5)
    # run_1(0.6)
    # run_1(0.6)
    # run_1(0.6)