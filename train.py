from email import generator
from new_data_zwh.dataset import DividedDataset
from eval import eval_model
from torch.utils.data import DataLoader,random_split

import os
import logging
from utils.functional import get_time_str
import torch
import torch.nn as nn
from torch import optim
import argparse
from torch.utils.tensorboard import SummaryWriter
from model.utnet import UTNet
from tqdm import tqdm

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by x every y epochs"""
    lr =lr * (0.5 ** (epoch// 5))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train_model(
    model,
    trainset,
    valset,
    device,
    epochs=100,
    batch_size=1,
    lr=0.00001,
    save_checkpoint=True,
    checkpoint_dir='checkpoints/',
    dice_only='true',
    ntimesfold=1
):  
    # n_train,n_val=157,15633
    # train_set, val_set = random_split(trainset, [n_train, n_val],generator=torch.Generator().manual_seed(42))
    train_loader=DataLoader(trainset,batch_size=batch_size,shuffle=True)
    writer = SummaryWriter(comment=f'{ntimesfold}th_BS_{batch_size}_LR_{lr}')
    global_step=0

    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=1e-4,momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    highest_dice=0
    for epoch in range(epochs):

        epoch_loss=0
        with tqdm(total=len(train_set),desc=f'Epoch {epoch+1}/{epochs}/{ntimesfold}th',unit='slice',ncols=120) as pbar:
            for batch in train_loader:
                imgs=batch['image']
                masks_true=batch['mask']

                imgs = imgs.to(device=device, dtype=torch.float32)
                masks_true = masks_true.to(device=device, dtype=torch.float32)

                pred=model(imgs)
                loss = criterion(pred, torch.squeeze(masks_true.long(),dim=1))

                epoch_loss+=loss.item()

                writer.add_scalar('Loss/train', loss.item(), global_step)
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                adjust_learning_rate(optimizer, epoch, lr)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step+=1
                

        score=eval_model(model=model,device=device,dataset=valset,batch_size=batch_size,desc='Evaluate')
        dice=score['dice']
        logging.info(f'dice is {dice}')
        writer.add_scalar('dice', dice, global_step)

        #保存最后一个batch查看
        writer.add_images('masks/true', masks_true, global_step)
        sf=nn.Softmax(dim=1)
        pred=(sf(pred)>0.5).float()
        #pred=torch.as_tensor(pred)
        writer.add_images('masks/imgs',imgs , global_step)
        writer.add_images('masks/pred0',torch.unsqueeze(pred[:,0,:,:],dim=1) , global_step)
        writer.add_images('masks/pred1',torch.unsqueeze(pred[:,1,:,:],dim=1) , global_step)
        
        if dice_only.lower()=='true':
            if dice>=highest_dice:
                if save_checkpoint:
                    if not os.path.exists(checkpoint_dir):
                        try:
                            os.makedirs(checkpoint_dir)
                            logging.info('Checkpoint directory is created.')
                        except OSError:
                            pass
                    torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'CP_{get_time_str()}_epoch{epoch+1}_{ntimesfold}_result{dice}.pth'))
                    logging.info(f'Checkpoint {epoch+1} of {ntimesfold} is saved.')
                    highest_dice=dice
            else:
                logging.info('Rubbish! Do not need to save!')
        else:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'CP_{get_time_str()}_epoch{epoch+1}_result{dice}.pth'))
            logging.info(f'Checkpoint {epoch+1} of {ntimesfold}is saved.')

    writer.close()

def get_args():
    parser = argparse.ArgumentParser(description='Train models')

    parser.add_argument('--cpu', dest='cpu', action='store_true',
                        default=False, help='Use cpu')
    parser.add_argument('-b', '--batchsize', dest='batchsize', type=int, nargs='?',
                        default=16, help='Batch size')
    parser.add_argument('-e', '--epochs', dest='epochs', type=int,
                        default=20, help='Number of epochs')
    parser.add_argument('-f', '--load', dest='load', type=str,
                        default=False, help='Load model from a .pth file')
    parser.add_argument('-l', '--lr', dest='lr', type=float, nargs='?',
                        default=0.05, help='Learning rate')
    parser.add_argument('-s', '--save', dest='save', type=str,
                        default='checkpoints/', help='Dir path to save .pth file')
    parser.add_argument('-d', '--datadir', dest='datadir', type=str,
                        default='new_data_zwh/', help='Dir path to find data')
    parser.add_argument('-k', '--dividedir', dest='dividedir', type=str,
                        default='new_data_zwh/divide', help='Dir path to find data')
    parser.add_argument( '--logname', dest='logname', type=str,
                        default='logs/log.txt', help='Dir path to find data')

    return parser.parse_args()

if __name__=='__main__':
    args = get_args()

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                        filename=args.logname,
                        filemode='a')
    
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    logging.info(f'Device: {device}')

    data_dir = args.datadir
    divide_dir = args.dividedir
    
    train_dir = os.path.join(divide_dir, 'train/')
    val_dir = os.path.join(divide_dir, 'val/')

    train_list=os.listdir(train_dir)
    train_list.sort()

    for n in train_list:
        logging.info(f'Now is the {n}')

        #parameters!
        model = UTNet(
        in_chan=1, 
        base_chan=32,
        num_classes=2, 
        reduce_size=16, 
        block_list='1234', 
        num_blocks=[1,1,1,1], 
        num_heads=[4,4,4,4], 
        projection='interp', 
        attn_drop=0.1, 
        proj_drop=0.1, 
        rel_pos=True, 
        aux_loss=False, 
        maxpool=True
        )
    
        if args.load:
            model.load_state_dict(torch.load(args.load, map_location=device))
            logging.info(f'Model is loaded from {args.load}.')

        model.to(device=device)
        fold_train = os.path.join(train_dir, n)
        fold_val = os.path.join(val_dir, n)
        train_set = DividedDataset(data_dir, fold_train, is_train=True)
        val_set = DividedDataset(data_dir, fold_val, is_train=False)

        train_model(
            model,
            trainset=train_set,
            valset=val_set,
            device=device,
            epochs=args.epochs,
            batch_size=args.batchsize,
            lr=args.lr,
            save_checkpoint=True,
            checkpoint_dir=args.save,
            dice_only='true',
            ntimesfold=n
        )









                

                
        
            

