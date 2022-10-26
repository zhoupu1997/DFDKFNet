import torch
import torch.nn.functional as F
from tqdm import tqdm

from dice_loss import dice_coeff


def eval(net, loader, device, n_val):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='img', leave=False) as pbar:
        for batch in loader:
            imgs = batch['image']
            true_masks = batch['mask']
            ndwi = batch['ndwi']
            ndvi = batch['ndvi']

            imgs = imgs.to(device=device, dtype=torch.float32)
            mask_type = torch.float32 if net.num_classes == 1 else torch.long
            true_masks = true_masks.to(device=device, dtype=mask_type)
            ndwi = ndwi.to(device=device, dtype=torch.float32)
            ndvi = ndvi.to(device=device, dtype=torch.float32)
            mask_pred = net(imgs, ndwi, ndvi)
            for true_mask, pred in zip(true_masks, mask_pred):
                pred = (pred > 0.5).float()  ##这里的0.5为threshold, 与预测时的threshold一样
                if net.n_classes > 1:
                    tot += F.cross_entropy(pred.unsqueeze(dim=0), true_mask.unsqueeze(dim=0).squeeze(1)).item()
                else:
                    tot += dice_coeff(pred, true_mask.squeeze(dim=1)).item()
            pbar.update(imgs.shape[0])

    return tot / n_val
