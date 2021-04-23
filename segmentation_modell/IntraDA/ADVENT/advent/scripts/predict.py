import sys
import pdb
from PIL import Image
import argparse
import os
import os.path as osp
import pprint
import warnings
import cv2
from torch.utils import data
import numpy as np
from advent.model.deeplabv2 import get_deeplab_v2
import torch
import torchvision
from torchvision import transforms
import torch.nn.functional as F
from torchvision.utils import make_grid
from torch import nn
colormap = [[0,0,0],[128,0,0],[0,128,0],[128,128,0]]
def create_seg(labels,colormap_):
    rgb_list = []
    for label_mask in labels:
        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for c in range(0,len(colormap_)):
            r[label_mask==c] = colormap_[c][0]
            g[label_mask==c] = colormap_[c][1]
            b[label_mask==c] = colormap_[c][2]
        rgb = np.zeros((label_mask.shape[0],label_mask.shape[1],3))
        rgb[:,:,0] = r/255.0
        rgb[:,:,1] = g/255.0
        rgb[:,:,2] = b/255.0
        rgb_list.append(rgb)   
    return torch.from_numpy(np.array(rgb_list).transpose([0, 3, 1, 2]))


def main():
    # LOAD ARGS
    parser = argparse.ArgumentParser(description="Code for prediction")
    parser.add_argument('--num_classes',default=4)
    parser.add_argument('--model',type=str,default='DeepLabv2')
    parser.add_argument('--checkpoint', type=str, default='C:/semseg/IntraDA/ADVENT/pretrained_models/model_120000.pth',
                        help='path to checkpoint', )
    parser.add_argument('--image_root', type=str, default='C:/semseg/IntraDA/ADVENT/sample',
                        help="root path to images for predicting")
    parser.add_argument('--gpuids', default=0)
    args = parser.parse_args()
    # load models
    if args.model == 'DeepLabv2':
        model = get_deeplab_v2(num_classes=args.num_classes,
                                multi_level=True)
    else:
        raise NotImplementedError(f"Not yet supported")
    if os.environ.get('ADVENT_DRY_RUN', '0') == '1':
        return
    saved_state_dict = torch.load(args.checkpoint,map_location='cpu')
    model.load_state_dict(saved_state_dict)
    
    device = args.gpuids
    model.eval()
    model.to(device)
    imgs = os.listdir(args.image_root)
    print("loaded images successfully")
    interp = nn.Upsample(size=(1024, 1280), mode='bilinear',
                         align_corners=True)
    for i in imgs:
        img = Image.open(args.image_root+'/'+i)
        image = img 
        img = img.convert('RGB')
        img = img.resize((1280, 1024), Image.BICUBIC)
        img = np.asarray(img,np.float32)
        img = img[:, :, ::-1]  # change to BGR
        img -= (128, 128, 128)
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy()).unsqueeze(0)
        print(img.shape)
        pred_main,_ = model(img.cuda(device))
        _,label = torch.max(interp(pred_main),1)
        #grid_image = make_grid(torch.from_numpy(np.array(colorize_mask(np.asarray(
        #np.argmax(F.softmax(pred_main).cpu().data[0].numpy().transpose(1, 2, 0),
        #          axis=2), dtype=np.uint8)).convert('RGB')).transpose(2, 0, 1)), 3,
         #                  normalize=False, range=(0, 255))
        #_,label = torch.max(pred_main,1)
        grid_image = torchvision.utils.make_grid(create_seg(label.detach().cpu().numpy(),colormap),3, normalize=False, range=(0, 255))
        #grid_image.show()
        torchvision.utils.save_image(grid_image,"./output/"+"{}_mask.png".format(i[0:-4]))
        #print(grid_image.shape)
        #print(pred_main.shape)        
def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    palette = label_colormap()
    new_mask.putpalette(palette.flatten())
    return new_mask

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
if __name__ == '__main__':
    main()
