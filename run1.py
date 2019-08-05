import os
import sys

def pruning_scratch_resnet20(gpu:int, rate_norm:float, 
                             rate_simi:float): 
    command = 'CUDA_VISIBLE_DEVICES={0} python  pruning_cifar12.py  \
    ./data/cifar.python --dataset cifar10 --arch resnet20 \
    --save_path ./result/scrath/cifar10_res20_norm{1:.2f}_simi{2:.2f} \
    --epochs 200 \
    --schedule 60 120 160 \
    --gammas 0.2 0.2 0.2 \
    --learning_rate 0.1 --decay 0.0005 --batch_size 128 \
    --rate_norm {1} --rate_simi {2} \
    --fuse 0.8 --simi_type cos --simi_ch ikk --mode merge\
    --layer_begin 0  --layer_end 54 --epoch_prune 1'.format(gpu, rate_norm, rate_simi)
    # print(command)
    return command

def run_1():
    command = pruning_scratch_resnet20(gpu=0, 
                                       rate_norm=1.0,
                                       rate_simi=0.3,
                                       )
    os.system(command)


if __name__ == "__main__":
    run_1()