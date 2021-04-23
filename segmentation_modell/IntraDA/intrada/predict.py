import sys
from tqdm import tqdm
import argparse
import os
import os.path as osp
import pprint
import torch
import numpy as np
from PIL import Image
from torch import nn
from torch.utils import data
from advent.model.deeplabv2 import get_deeplab_v2
from advent.model.discriminator import get_fc_discriminator
from advent.dataset.cityscapes import CityscapesDataSet
from advent.utils.func import prob_2_entropy
import torch.nn.functional as F
from advent.utils.func import loss_calc, bce_loss
from advent.domain_adaptation.config import cfg, cfg_from_file
from matplotlib import pyplot as plt
from matplotlib import image  as mpimg
import torch


def label_colormap(n_label=256, value=None):
    """Label colormap.
    Parameters
    ----------
    n_labels: int
        Number of labels (default: 256).
    value: float or int
        Value scale or value of label color in HSV space.
    Returns
    -------
    cmap: numpy.ndarray, (N, 3), numpy.uint8
        Label id to colormap.
    """

    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    cmap = np.zeros((n_label, 3), dtype=np.uint8)
    for i in range(0, n_label):
        id = i
        r, g, b = 0, 0, 0
        for j in range(0, 8):
            r = np.bitwise_or(r, (bitget(id, 0) << 7 - j))
            g = np.bitwise_or(g, (bitget(id, 1) << 7 - j))
            b = np.bitwise_or(b, (bitget(id, 2) << 7 - j))
            id = id >> 3
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b

    if value is not None:
        hsv = color_module.rgb2hsv(cmap.reshape(1, -1, 3))
        if isinstance(value, float):
            hsv[:, 1:, 2] = hsv[:, 1:, 2].astype(float) * value
        else:
            assert isinstance(value, int)
            hsv[:, 1:, 2] = value
        cmap = color_module.hsv2rgb(hsv).reshape(-1, 3)
    return cmap


def colorize(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    palette = label_colormap().flatten()
    new_mask.putpalette(palette)
    
    return new_mask    

def colorize_save(output_pt_tensor, name):
    output_np_tensor = output_pt_tensor.cpu().data[0].numpy()
    mask_np_tensor   = output_np_tensor.transpose(1,2,0) 
    mask_np_tensor   = np.asarray(np.argmax(mask_np_tensor, axis=2), dtype=np.uint8)
    mask_Img         = Image.fromarray(mask_np_tensor)
    mask_color       = colorize(mask_np_tensor)  

    name = name.split('/')[-1]
    #mask_Img.save('./color_masks/%s' % (name))
    mask_color.save('./color_masks5/%s_color.png' % (name.split('.')[0]))


def main():
    parser = argparse.ArgumentParser("code for prediction")
    parser.add_argument('--gpu_id',type= str, default = 0)
    parser.add_argument('--checkpoint',type= str, default= '../ADVENT/experiments/snapshots/SimRunway2RealRunway_DeepLabv2_AdvEnt_5class/model_118000.pth')
    parser.add_argument('--images_path',type=str,default=r'\\RS3618\student\Fei Yin\Flugversuchsdaten\CP+CI2019-08-06-14-19-27-LOAN27')
    parser.add_argument('--normalize',type=bool, default=False)
    parser.add_argument('--cfg', type=str, default='../ADVENT/advent/scripts/configs/advent.yml',
                        help='optional config file' )
    args = parser.parse_args()
    assert args.cfg is not None, 'Missing cfg file'
    cfg_from_file(args.cfg)
    # create model and load checkpoint

    model_gen = get_deeplab_v2(num_classes=cfg.NUM_CLASSES, multi_level=cfg.TEST.MULTI_LEVEL)
    
    #restore_from = osp.join(cfg.TEST.SNAPSHOT_DIR[0], f'model_{args.best_iter}.pth')
    restore_from = args.checkpoint
    print("Loading the generator:", restore_from)
    device = args.gpu_id
    saved_state_dict = torch.load(restore_from)
    model_gen.load_state_dict(saved_state_dict)
    model_gen.eval()
    model_gen.cuda(device)

    interp_target = nn.Upsample(size=(1024, 1280), mode='bilinear',
                                align_corners=True)
    # get images
    root = args.images_path
    for i in os.listdir(root):
        if os.path.isfile(os.path.join(root,i)):
            img = Image.open(osp.join(root,i))
            img = img.convert('RGB')
            img = img.resize((1024, 512), Image.BICUBIC)
            img = np.asarray(img, np.float32)
            img = img[:, :, ::-1]  # change to BGR
            img -= np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
            img = img.transpose((2, 0, 1))
            image = torch.from_numpy(img.copy())
            image_batch = torch.unsqueeze(image,0)
            print(image.shape)
            print(image_batch.shape)
            # prediction
            with torch.no_grad():
                _, pred_trg_main = model_gen(image_batch.cuda(device))
                pred_trg_main    = interp_target(pred_trg_main)
                if args.normalize == True:
                    normalizor = (11-len(find_rare_class(pred_trg_main))) / 11.0 + 0.5
                else:
                    normalizor = 1
                #pred_trg_entropy = prob_2_entropy(F.softmax(pred_trg_main))
                #entropy_list.append((name[0], pred_trg_entropy.mean().item() * normalizor))
                colorize_save(pred_trg_main, i.split('.')[0]+'.png')


if __name__ == "__main__":
    main()