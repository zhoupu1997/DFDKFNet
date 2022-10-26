from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
import cv2
from PIL import Image
from osgeo import gdal
import torchvision.transforms as transforms

gdal.PushErrorHandler('CPLQuietErrorHandler')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Dataset_(Dataset):
    def __init__(self, imgs_dir, masks_dir, ndvi_dir, ndwi_dir, merge_band=True):

        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.ndvi_dir = ndvi_dir
        self.ndwi_dir = ndwi_dir
        self.merge_band = merge_band

        self.img_ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        self.label_ids = [splitext(file)[0] for file in listdir(masks_dir)
                        if not file.startswith('.')]
        self.ndvi_ids = [splitext(file)[0] for file in listdir(ndvi_dir)
                        if not file.startswith('.')]
        self.ndwi_ids = [splitext(file)[0] for file in listdir(ndwi_dir)
                         if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.img_ids)} examples')

    def __len__(self):
        return len(self.img_ids)

    def preprocess(self, pil_img):

        img_nd = np.array(pil_img)
        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)
        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        return img_trans

    def img_read(self, img_path):
        dataset_img = gdal.Open(img_path)
        Tif_width = dataset_img.RasterXSize  # 栅格矩阵的列数
        Tif_height = dataset_img.RasterYSize  # 栅格矩阵的行数
        img = dataset_img.ReadAsArray(0, 0, Tif_width, Tif_height)
        return img

    def __getitem__(self, i, merge_band=True):   ##

        # 水体提取的getitem
        merge_band = self.merge_band
        img_idx = self.img_ids[i]
        label_idx = self.label_ids[i]
        ndvi_idx = self.ndvi_ids[i]
        ndwi_idx = self.ndwi_ids[i]
        img_read = self.img_read

        mask_file = glob(self.masks_dir + label_idx + '.tif')
        img_file = glob(self.imgs_dir + img_idx + '.tif')
        ndvi_file = glob(self.ndvi_dir + ndvi_idx + '.tif')
        ndwi_file = glob(self.ndwi_dir + ndwi_idx + '.tif')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {label_idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {img_idx}: {img_file}'
        assert len(ndvi_file) == 1, \
            f'Either no image or multiple images found for the ID {ndvi_idx}: {ndvi_file}'
        assert len(ndwi_file) == 1, \
            f'Either no image or multiple images found for the ID {ndwi_idx}: {ndwi_file}'

        # img的读取   每次先看一下数据的范围
        img = img_read(img_file[0])
        img = img / 255.0
        # print("img:", img)
        mask = img_read(mask_file[0])
        mask = mask / 255.0
        # print("mask:", mask)
        ndvi = img_read(ndvi_file[0])       ##这里的ndvi和ndwi的取值范围在0~1之间
        ndvi = ndvi / 255.0
        ndwi = img_read(ndwi_file[0])       #这里将不在0~1之间的值变为0
        ndwi = ndwi / 255.0
        mask = self.preprocess(mask)   ##[H, W]=>[1, H, W]
        ndvi = self.preprocess(ndvi)   ##[H, W]=>[1, H, W]
        ndwi = self.preprocess(ndwi)   ##[H, W]=>[1, H, W]
        img = np.concatenate((img, ndvi, ndwi), 0)

        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask), 'ndwi': torch.from_numpy(ndwi), 'ndvi': torch.from_numpy(ndvi)}   #把数组转换为张量