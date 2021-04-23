# demo.py
#
import cv2
import argparse
import os
import numpy as np
import time
from mypath import Path

from modeling.deeplab import *
from dataloaders import custom_transforms as tr
from PIL import Image
from torchvision import transforms
from dataloaders.utils import  *
from torchvision.utils import make_grid, save_image

def adjust_gamma(image, gamma=1.0):
    lookUpTable = np.empty((1, 256), np.uint8)
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    res = cv2.LUT(image, lookUpTable)
    res = cv2.cvtColor(res,cv2.COLOR_BGR2RGB)
    return res

def alpha_beta(image, alpha, beta):
    result = cv2.convertScaleAbs(image, beta, alpha)
    return result


def replaceZeroes(data):
    min_nonzero = min(data[np.nonzero(data)])
    data[data == 0] = min_nonzero
    return data

def SSR(src_img, size):
    L_blur = cv2.GaussianBlur(src_img, (size, size), 0)
    img = replaceZeroes(src_img)
    L_blur = replaceZeroes(L_blur)

    dst_Img = cv2.log(img/255.0)
    dst_Lblur = cv2.log(L_blur/255.0)
    dst_IxL = cv2.multiply(dst_Img,dst_Lblur)
    log_R = cv2.subtract(dst_Img, dst_IxL)

    dst_R = cv2.normalize(log_R,None,0,255,cv2.NORM_MINMAX)
    log_uint8 = cv2.convertScaleAbs(dst_R)
    return log_uint8

def replaceZeroes(data):
    min_nonzero = min(data[np.nonzero(data)])
    data[data == 0] = min_nonzero
    return data


def MSR(img, scales):
    weight = 1 / 3.0
    scales_size = len(scales)
    h, w = img.shape[:2]
    log_R = np.zeros((h, w), dtype=np.float32)

    for i in range(scales_size):
        img = replaceZeroes(img)
        L_blur = cv2.GaussianBlur(img, (scales[i], scales[i]), 0)
        L_blur = replaceZeroes(L_blur)
        dst_Img = cv2.log(img/255.0)
        dst_Lblur = cv2.log(L_blur/255.0)
        dst_Ixl = cv2.multiply(dst_Img, dst_Lblur)
        log_R += weight * cv2.subtract(dst_Img, dst_Ixl)

    dst_R = cv2.normalize(log_R,None, 0, 255, cv2.NORM_MINMAX)
    log_uint8 = cv2.convertScaleAbs(dst_R)
    return log_uint8
def adapthiseq(image,clipLimit=2.0, tileGridSize=(8, 8)):
    B, G, R = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit, tileGridSize)
    cl1_B = clahe.apply(B)
    cl1_G = clahe.apply(G)
    cl1_R = clahe.apply(R)
    cl1 = cv2.merge((cl1_R, cl1_G, cl1_B))
    return cl1
def Pixel_Accuracy(confusion_matrix):
        Acc = np.diag(confusion_matrix).sum() / confusion_matrix.sum()
        return Acc

def Pixel_Accuracy_Class(confusion_matrix):
    Acc = np.diag(confusion_matrix) / confusion_matrix.sum(axis=1)
    Acc = np.nanmean(Acc)
    return Acc

def Mean_Intersection_over_Union(confusion_matrix):
    MIoU = np.diag(confusion_matrix) / (
                np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
                np.diag(confusion_matrix))
    #MIoU = np.nanmean(MIoU)
    return MIoU

def Frequency_Weighted_Intersection_over_Union(confusion_matrix):
    freq = np.sum(confusion_matrix, axis=1) / np.sum(confusion_matrix)
    iu = np.diag(confusion_matrix) / (
                np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
                np.diag(confusion_matrix))

    FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
    return FWIoU

def _generate_matrix(num_class,gt_image, pre_image):
    mask = (gt_image >= 0) & (gt_image < num_class)
    label = num_class * gt_image[mask].astype('int') + pre_image[mask]
    count = np.bincount(label, minlength=num_class**2)
    confusion_matrix = count.reshape(num_class, num_class)
    return confusion_matrix

def add_batch(num_class,confusion_matrix, gt_image, pre_image):
    assert gt_image.shape == pre_image.shape
    confusion_matrix += _generate_matrix(num_class,gt_image, pre_image)

def reset(confusion_matrix,num_class):
    confusion_matrix = np.zeros((num_class,) * 2)

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
def main():

    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--in-path',type=str, required=True, help='image to test')
    # parser.add_argument('--out-path', type=str, required=True, help='mask image to save')
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--ckpt', type=str, default='deeplab-resnet.pth',
                        help='saved model')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['pascal', 'coco', 'cityscapes','invoice'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--crop-size', type=int, default=513,
                        help='crop image size')
    parser.add_argument('--num_classes', type=int, default=4,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False
    model_s_time = time.time()
    model = DeepLab(num_classes=args.num_classes,
                    backbone=args.backbone,
                    output_stride=args.out_stride,
                    sync_bn=args.sync_bn,
                    freeze_bn=args.freeze_bn)
    
    ckpt = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    model = model.cuda()
    model_u_time = time.time()
    model_load_time = model_u_time-model_s_time
    print("model load time is {}".format(model_load_time))

    composed_transforms = transforms.Compose([
        tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        tr.ToTensor()])
    images = os.listdir(os.path.join(args.in_path,'images','val1'))
    labels = os.listdir(os.path.join(args.in_path,'labels','train'))
    confusion_matrix = np.zeros((args.num_classes,) * 2)
    reset(confusion_matrix,args.num_classes)
    for index, (img, lbl) in enumerate(zip(images,labels)):
        s_time = time.time()
        ####### image enhancement ##########
        '''
        image_ = args.in_path+"/"+'images'+'/'+'val1'+'/'+img
        scales = [15, 101, 301]
        src_img = cv2.imread(image_)
        b_gray, g_gray, r_gray = cv2.split(src_img)
        b_gray = MSR(b_gray, scales)
        g_gray = MSR(g_gray, scales)
        r_gray = MSR(r_gray, scales)
        
        #image = cv2.merge([r_gray, g_gray, b_gray])
        
        
        result = cv2.merge([b_gray, g_gray, r_gray])
        image = adapthiseq(result,2)
        '''
        ###adaptive histogramm equalization
        '''
        image_ = args.in_path+"/"+'images'+'/'+'val1'+'/'+img
        src_img = cv2.imread(image_)
        image = adapthiseq(src_img,2)
        '''
        ###gamma conversion
        '''
        image_ = args.in_path+"/"+'images'+'/'+'val1'+'/'+img
        src_img = cv2.imread(image_)
        image = adjust_gamma(src_img,0.4)
        '''
        ###alpah beta
        '''
        image_ = args.in_path+"/"+'images'+'/'+'val1'+'/'+img
        src_img = cv2.imread(image_)
        res = alpha_beta(src_img, 2, 40)
        image = cv2.cvtColor(res,cv2.COLOR_BGR2RGB)
        '''
        ###SSR
        '''
        image_ = args.in_path+"/"+'images'+'/'+'val1'+'/'+img
        src_img = cv2.imread(image_)
        size=3
        b_gray, g_gray, r_gray = cv2.split(src_img)
        b_gray = SSR(b_gray, size)
        g_gray = SSR(g_gray, size)
        r_gray = SSR(r_gray, size)
        image = cv2.merge([r_gray, g_gray, b_gray])
        '''
        '''
        result = cv2.merge([b_gray, g_gray, r_gray])
        image = adapthiseq(result,2)
        '''
        #image = cv2.merge([r_gray, g_gray, b_gray])
        
        ####### image enhancement ##########
        image = Image.open(os.path.join(args.in_path,'images','val1',img)).convert('RGB')
        label = Image.open(os.path.join(args.in_path,'labels','train',lbl))
        sample = {'image': image, 'label': label}
        tensor_in = composed_transforms(sample)['image'].unsqueeze(0)

        model.eval()
        if args.cuda:
            tensor_in = tensor_in.cuda()
        with torch.no_grad():
            output = model(tensor_in)
        pred = output.data.cpu().numpy()
        target = composed_transforms(sample)['label'].unsqueeze(0)
        target = target.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        add_batch(args.num_classes,confusion_matrix,target,pred)
        '''
        confusion_matrix = _generate_matrix(args.num_classes,target,pred)
        #Acc = Pixel_Accuracy(confusion_matrix)
        #Acc_class = Pixel_Accuracy_Class(confusion_matrix)
        mIoU = Mean_Intersection_over_Union(confusion_matrix)
        #FWIoU = Frequency_Weighted_Intersection_over_Union(confusion_matrix)
        #print(mIoU)
        '''
        '''
        with open('./10295_enhance.txt','a+') as f:
            f.write(' '.join(map(str,mIoU)))
            f.write('\n')
        f.close()
        '''
    
    Acc = Pixel_Accuracy(confusion_matrix)
    Acc_class = Pixel_Accuracy_Class(confusion_matrix)
    mIoU = Mean_Intersection_over_Union(confusion_matrix)
    FWIoU = Frequency_Weighted_Intersection_over_Union(confusion_matrix)
    print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
    
if __name__ == "__main__":
   main()
