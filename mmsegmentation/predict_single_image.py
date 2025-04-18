from mmseg.apis import inference_segmentor, init_segmentor
import mmcv
import glob
import os
from tqdm import tqdm
import argparse
from PIL import Image
import numpy as np 
from skimage.io import imsave
import pdb



parser = argparse.ArgumentParser(description="")
parser.add_argument("--config_file", default='./work_dirs/upernet_swin_base_patch4_window12_512x512_160k_egohos_handobj2_pretrain_480x360_22K/upernet_swin_base_patch4_window12_512x512_160k_egohos_handobj2_pretrain_480x360_22K.py', type=str)
parser.add_argument("--checkpoint_file", default='./work_dirs/upernet_swin_base_patch4_window12_512x512_160k_egohos_handobj2_pretrain_480x360_22K/best_mIoU_iter_42000.pth', type=str)
parser.add_argument("--img_path", default='../data/train/image/xxx.png', type=str)
parser.add_argument("--pred_seg_dir", default='./work_dirs/upernet_swin_base_patch4_window12_512x512_160k_egohos_handobj2_pretrain_480x360_22K/outputs/train_seg', type=str)
args = parser.parse_args()

os.makedirs(args.pred_seg_dir, exist_ok = True)

# build the model from a config file and a checkpoint file
model = init_segmentor(args.config_file, args.checkpoint_file, device='cuda:0')

alpha = 0.5
img = np.array(Image.open(args.img_path))
seg_result = inference_segmentor(model, args.img_path)[0]
# print(seg_result.shape)
# print(seg_result.max())
# print(np.unique(seg_result))
fname = os.path.basename(args.img_path).split('.')[0]
imsave(os.path.join(args.pred_seg_dir, fname + '.png'), seg_result.astype(np.uint8))

