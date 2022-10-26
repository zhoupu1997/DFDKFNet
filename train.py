import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from pgunet.FCN import FCN8s_1, VGGNet, FCN8s, FCN32s
from DFDKFNet import DFDKFNet_
from HRNet import HighResolutionNet
from loss_0304 import Weight_Lossv1
from tensorboardX import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from eval_net import eval
from Datasets import Dataset_
from losses import dice_coeff
from torch import randperm
from torch.nn import init


dir_img = r"C:\Users\user\Desktop\experiment_dataset_grsl\google_aug/"
dir_mask = r"C:\Users\user\Desktop\experiment_dataset_grsl\mask_aug/"
dir_ndvi = r"C:\Users\user\Desktop\experiment_dataset_grsl\ndvi_aug/"
dir_ndwi = r"C:\Users\user\Desktop\experiment_dataset_grsl\ndwi_aug/"
dir_checkpoint1 = r"C:\Users\user\Desktop\experiment_dataset_grsl\ckpt\ckpt_DFDKFNet_aug_cross_validation_1/"
dir_checkpoint2 = r"C:\Users\user\Desktop\experiment_dataset_grsl\ckpt\ckpt_DFDKFNet_aug_cross_validation_2/"
dir_checkpoint3 = r"C:\Users\user\Desktop\experiment_dataset_grsl\ckpt\ckpt_DFDKFNet_aug_cross_validation_3/"
dir_checkpoint4 = r"C:\Users\user\Desktop\experiment_dataset_grsl\ckpt\ckpt_DFDKFNet_aug_cross_validation_4/"
dir_checkpoint5 = r"C:\Users\user\Desktop\experiment_dataset_grsl\ckpt\ckpt_DFDKFNet_aug_cross_validation_5/"


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=201,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=12,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=20,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    # 也可以判断是否为conv2d，使用相应的初始化方式
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
     # 是否为批归一化层
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

if __name__ == '__main__':


    dataset = Dataset(dir_img, dir_mask, dir_ndvi, dir_ndwi, merge_band=True)
    print(len(dataset))
    lenth = randperm(len(dataset), generator=torch.Generator().manual_seed(0)).tolist()  # 生成乱序的索引
    rand_dataset = torch.utils.data.Subset(dataset, lenth)   ##利用乱序的索引将数据集打乱
    indices1 = range(0, 4537)  # 取标号为第0个到第4537个数据
    indices2 = range(4537, 9074)
    indices3 = range(9074, 13611)
    indices4 = range(13611, 18148)
    indices5 = range(18148, 22686)
    #交叉验证
    train1 = torch.utils.data.Subset(rand_dataset, list(indices2) + list(indices3) + list(indices4) + list(indices5))
    n_train1 = len(train1)
    val1 = torch.utils.data.Subset(rand_dataset, indices1)
    n_val1 = len(val1)
    #
    train2 = torch.utils.data.Subset(rand_dataset, list(indices1) + list(indices3) + list(indices4) + list(indices5))
    n_train2 = len(train2)
    val2 = torch.utils.data.Subset(rand_dataset, indices2)
    n_val2 = len(val2)
    #
    train3 = torch.utils.data.Subset(rand_dataset, list(indices2) + list(indices1) + list(indices4) + list(indices5))
    n_train3 = len(train3)
    val3 = torch.utils.data.Subset(rand_dataset, indices3)
    n_val3 = len(val3)
    #
    train4 = torch.utils.data.Subset(rand_dataset, list(indices2) + list(indices3) + list(indices1) + list(indices5))
    n_train4 = len(train4)
    val4 = torch.utils.data.Subset(rand_dataset, indices4)
    n_val4 = len(val4)
    #
    train5 = torch.utils.data.Subset(rand_dataset, list(indices2) + list(indices3) + list(indices4) + list(indices1))
    n_train5 = len(train5)
    val5 = torch.utils.data.Subset(rand_dataset, indices5)
    n_val5 = len(val5)


    for i in range(4,5):
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        args = get_args()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Using device {device}')
        net = DFDKFNet(n_channels=5, n_classes=1)
        net.apply(weight_init)

       
        if args.load:
            net.load_state_dict(
                torch.load(r"C:\Users\user\Desktop\plante_train\ckpt\ckpt_DFDKFNet\CP_epoch399.pth",
                           map_location=device)
            )
            logging.info(f'Model loaded from {args.load}')
        net.to(device=device)

        epochs = args.epochs
        batch_size = args.batchsize
        lr = args.lr
        device = device
        img_scale = args.scale
        val_percent = args.val / 100
        save_cp = True


        train_loader = DataLoader(eval("train" + str(i+1)), batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        val_loader = DataLoader(eval("val" + str(i+1)), batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        #这里的eval是将字符串变为赋值的变量
        writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
        global_step = 0

        logging.info(f'''Starting training:
                Folds:           {i}
                Epochs:          {epochs}
                Batch size:      {batch_size}
                Learning rate:   {lr}
                Training size:   {eval("n_train"+str(i+1))}
                Validation size: {eval("n_val"+str(i+1))}
                Checkpoints:     {save_cp}
                Device:          {device.type}
                Images scaling:  {img_scale}
            ''')
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)  # weight_decay L2正则化的系数

        criterion = nn.BCELoss()

        for epoch in range(epochs):
            net.train()
            epoch_loss = 0
            # 进度条配置，total为总的迭代次数；desc为进度条的前缀；unit为每个迭代的单元
            with tqdm(total=eval("n_train"+str(i+1)), desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
                for batch in train_loader:
                    imgs = batch['image']
                    true_masks = batch['mask']
                    ndwi = batch['ndwi']
                    ndvi = batch['ndvi']
                    # print(imgs.shape)
                    # print(true_masks.shape)

                    imgs = imgs.to(device=device, dtype=torch.float32)  # 将imgs和mask放到device上（GPU上）
                    mask_type = torch.float32 if net.n_classes == 1 else torch.long
                    true_masks = true_masks.to(device=device, dtype=mask_type)
                    ndwi = ndwi.to(device=device, dtype=torch.float32)
                    ndvi = ndvi.to(device=device, dtype=torch.float32)
                    masks_pred = net(imgs, ndwi, ndvi)

                    if net.n_classes == 1:
                        masks_pred = masks_pred.squeeze(1)
                        true_masks = true_masks.squeeze(1)
                    else:
                        masks_pred = masks_pred
                        true_masks = true_masks

                    loss = criterion(masks_pred, true_masks)
                    epoch_loss += loss.item()  # 一个元素张量可以用item得到元素值

                    loss = loss.cpu()
                    loss_ = str(loss.data.numpy())
                    with open('./loss_DFDKFNet_aug_cross_validation_{}.txt'.format(i+1), 'a',
                              newline='') as f:  ##这里的newline=''相当于删掉txt里面的空行
                        f.write(str(global_step))
                        f.write(' ')
                        f.write(loss_)
                        if global_step < 9999999:
                            f.write(' \r\n')

                    pbar.set_postfix(**{'loss (batch)': loss.item()})  ##输入一个字典，显示实验指标

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    pbar.update(imgs.shape[0])
                    global_step += 1  # 每过一个batch，global_step就+1
                    if global_step % (len(eval("train" + str(i+1))) // (1 * batch_size)) == 0:  # %是求余数， 这里是每一个batch计算一次val_score
                        val_score = eval_ndwi_ndvi_nodeep1(net, val_loader, device, eval("n_val" + str(i+1)))
                        print('val', epoch)  # 每一个batch，利用eval_net和val_loader来进行验证，从而得到每一个epoch的验证精度
                        if net.n_classes > 1:
                            logging.info('Validation cross entropy: {}'.format(val_score))
                            writer.add_scalar('Loss/test', val_score, epoch)

                        else:
                            logging.info('Validation Dice Coeff: {}'.format(val_score))
                            val_score_ = str(val_score)
                            with open('./valid_DFDKFNet_aug_cross_validation_{}.txt'.format(i+1), 'a', newline='') as f:
                                f.write(str(epoch))
                                f.write(' ')
                                f.write(val_score_)
                                if epoch < 9999999:
                                    f.write(' \r\n')
                            writer.add_scalar('Dice/test', val_score, epoch)  # 验证精度随epoch的变化情况
            if epoch % 1 == 0 or epoch == 199:
                if save_cp:
                    try:
                        os.mkdir(eval("dir_checkpoint" + str(i+1)))  # 这里是创建了一个目录，当dir_checkpoint不存在时，会在train_文件夹下面创建一个checkpoints文件夹
                        logging.info('Created checkpoint directory')
                    except OSError:
                        pass
                    torch.save(net.state_dict(),
                               eval("dir_checkpoint" + str(i+1)) + f'CP_epoch{epoch}.pth')
                    print('ckpt', epoch)
                    logging.info(f'Checkpoint {epoch} saved !')
                    print('done')

        writer.close()





