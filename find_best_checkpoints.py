import os
import re
import numpy
import shutil

def pick_bests(find_checkpoint, save_checkpoint):
    models = os.listdir(find_checkpoint)
    models.sort()

    zero,one,two,three,four=[[] for _ in range(5)],[[] for _ in range(5)],[[] for _ in range(5)],[[] for _ in range(5)],[[] for _ in range(5)]
    problem=[]

    pattern=r"result0.\d+"

    for model in models:
        if '_0_' in model:
            if '0.txt' in model:
                zero[0].append(model)
            if '1.txt' in model:
                zero[1].append(model)
            if '2.txt' in model:
                zero[2].append(model)
            if '3.txt' in model:
                zero[3].append(model)
            if '4.txt' in model:
                zero[4].append(model)
                
        elif '_1_' in model:
            if '0.txt' in model:
                one[0].append(model)
            if '1.txt' in model:
                one[1].append(model)
            if '2.txt' in model:
                one[2].append(model)
            if '3.txt' in model:
                one[3].append(model)
            if '4.txt' in model:
                one[4].append(model)

        elif '_2_' in model:
            if '0.txt' in model:
                two[0].append(model)
            if '1.txt' in model:
                two[1].append(model)
            if '2.txt' in model:
                two[2].append(model)
            if '3.txt' in model:
                two[3].append(model)
            if '4.txt' in model:
                two[4].append(model)
    
        elif '_3_' in model:
            if '0.txt' in model:
                three[0].append(model)
            if '1.txt' in model:
                three[1].append(model)
            if '2.txt' in model:
                three[2].append(model)
            if '3.txt' in model:
                three[3].append(model)
            if '4.txt' in model:
                three[4].append(model)

        elif '_4_' in model:
            if '0.txt' in model:
                four[0].append(model)
            if '1.txt' in model:
                four[1].append(model)
            if '2.txt' in model:
                four[2].append(model)
            if '3.txt' in model:
                four[3].append(model)
            if '4.txt' in model:
                four[4].append(model)
        else:
            problem.append(model)


    if  not os.path.exists(save_checkpoint):
        try:
            os.makedirs(save_checkpoint)
            print("Success")
        except OSError:
            print('Error')
    else:
        print('Exists')

    best_list=[]

    for i,item in enumerate(zero):
        max_v=max(item)
        max_p=item.index(max_v)
        best_list.append(zero[i][max_p])
    
    for i,item in enumerate(one):
        max_v=max(item)
        max_p=item.index(max_v)
        best_list.append(one[i][max_p])
    
    for i,item in enumerate(two):
        max_v=max(item)
        max_p=item.index(max_v)
        best_list.append(two[i][max_p])
    
    for i,item in enumerate(three):
        max_v=max(item)
        max_p=item.index(max_v)
        best_list.append(three[i][max_p])
    
    for i,item in enumerate(four):
        max_v=max(item)
        max_p=item.index(max_v)
        best_list.append(four[i][max_p])
    
    for file in best_list:
        src=os.path.join(find_checkpoint,file)
        dst=save_checkpoint
        shutil.copy(src, dst)

if __name__=='__main__':
    pick_bests(find_checkpoint='Utnet/checkpoints', save_checkpoint='Utnet/Utnet')