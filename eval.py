# -*- coding: utf-8 -*-
"""
Author: Zhang Wenhao @ asus

Created on 2021/11/14
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm

from utils.dice_score import dice_coeff_re
from utils.hausdorff_distance import modified_hau_dist_re


def eval_model(model: nn.Module, device, dataset, batch_size, in_training=True, desc='Evaluate'):
    model.eval()

    items_num = len(dataset)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    dice_score_sum = 0
    mhd_sum = 0

    dice_list = []
    mhd_list = []

    with tqdm(total=items_num, desc=desc, unit='slice',ncols=120) as pbar:

        for batch in loader:
            imgs, masks_true = batch['image'], batch['mask']

            imgs = imgs.to(device=device, dtype=torch.float32)
            masks_ture = masks_true.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                masks_pred = model(imgs)

                # sum dice of all slices
                masks_pred = torch.argmax(masks_pred, dim=1, keepdim=True).to(dtype=torch.float32)
                dice_score_sum += dice_coeff_re(masks_pred, masks_ture, reduction='sum')
                mhd_sum += modified_hau_dist_re(masks_pred, masks_ture, reduction='sum').item()
                
                # dice_list += dice_coeff_re(masks_pred, masks_ture, reduction='none')
                # mhd_list += modified_hau_dist_re(masks_pred, masks_ture, reduction='none').tolist()

            pbar.update(imgs.shape[0])

    dice_score = dice_score_sum / items_num
    m_hau_d = mhd_sum / items_num

    # result_df = pd.DataFrame({'dice': dice_list, 'mhd': mhd_list})
    # result_df.sort_values(by=['dice', 'mhd'], ascending=False, inplace=True)
    # print(result_df.std())

    if in_training:
        model.train()

    return {
        'dice': dice_score,
        'mhd': m_hau_d
    }
