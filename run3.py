import os
import sys

def pruning_scratch(gpu:int, renew:int, simi:float, pos:bool, simi_type:str, arch:str, mode='sup', ch='out'): 
    command = 'CUDA_VISIBLE_DEVICES={0} python  pruning_cifar12.py  \
    ./data/cifar.python --dataset cifar10 --arch {5} \
    --save_path ./result/scrath/{6}/cifar10/{5}/rn{1}_rate{2:.2f}_type{4}_pos{3:d}_{7} \
    --epochs 200 \
    --schedule 60 120 160 \
    --gammas 0.2 0.2 0.2 \
    --learning_rate 0.1 --decay 0.0005 --batch_size 128 \
    --simi_ch {7} --mode {6} --pos {3} --simi_type {4}  --rate_simi {2} \
    --layer_begin 0  --layer_end 54 --epoch_prune 1'.format(gpu, renew, simi, pos, simi_type, arch, mode, ch)
    print(command)
    return command

def run_1():
    command = pruning_scratch(gpu=0, renew=2, simi=0.4, pos=True, simi_type='l2', arch='resnet110', mode='sup')
    os.system(command)

def run_2():
    command = pruning_scratch(gpu=0, renew=2, simi=0, pos=True, simi_type='cos', arch='resnet110', mode='sup')
    os.system(command)

def run_3():
    command = pruning_scratch(gpu=0, renew=1, simi=0.3, pos=0, simi_type='l2', arch='resnet20', mode='sup')
    os.system(command)

def run_default():
    command = pruning_scratch(gpu=1, renew=2, simi=0, pos=0, simi_type='cos', arch='resnet110', mode='default')
    os.system(command)


simi_ls = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
pos_ls = [True, False]
renew_ls = [5, 10, 100]

def run_res20():
    for simi in simi_ls:
        command = pruning_scratch(gpu=1, renew=2, simi=simi, pos=True, simi_type='cos', arch='resnet20', mode='sup')
        os.system(command)

def run_res110():
    for simi in simi_ls:
        command = pruning_scratch(gpu=1, renew=2, simi=simi, pos=True, simi_type='cos', arch='resnet110', mode='sup')
        os.system(command)

if __name__ == "__main__":
    # run_1()
#     run_2()
    run_3()

    # run_default()
#     run_res20()
    # run_res110()
   