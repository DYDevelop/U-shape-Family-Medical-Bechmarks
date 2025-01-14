import os
import random
import argparse

from glob import glob
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default="busi", help='dataset_name')
parser.add_argument('--dataset_root', type=str, default="./data", help='dir')
args = parser.parse_args()

if __name__ == '__main__':

    name = args.dataset_name
    root = os.path.join(args.dataset_root, args.dataset_name)

    img_ids = glob(os.path.join(root, 'images', '*.png'))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=1225)

    with open(os.path.join(root, '{}_train.txt'.format(name)), 'w') as file:
        for i in train_img_ids:
            file.write(i + '\n')
    print("build train file successfully, path is: {}".format(os.path.join(root, '{}_train.txt'.format(name))))

    with open(os.path.join(root, '{}_val.txt'.format(name)), 'w') as file:
        for i in val_img_ids:
            file.writelines(i + '\n')
    print("build validate file successfully, path is: {}".format(os.path.join(root, '{}_val.txt'.format(name))))


# python split.py --dataset_root ../data --dataset_name axi
# python main.py --model U_Net --base_dir ../data/axi --train_file_dir axi_train.txt --val_file_dir axi_val.txt --base_lr 0.01 --epoch 300 --batch_size 8 --patience 10 --num_classes 2 --k_fold 5
# python main.py --model TransUnet --base_dir ../data/axi --train_file_dir axi_train.txt --val_file_dir axi_val.txt --base_lr 0.01 --epoch 300 --batch_size 8 --patience 10 --num_classes 2 --k_fold 5
# python main.py --model CMUNeXt --base_dir ../data/axi --train_file_dir axi_train.txt --val_file_dir axi_val.txt --base_lr 0.01 --epoch 300 --batch_size 8 --patience 10 --num_classes 2 --k_fold 5

# ["CMUNeXt", "CMUNet", "AttU_Net", "TransUnet", "R2U_Net", "U_Net", "UNext", "UNetplus", "UNet3plus", "SwinUnet", "MedT", "TransUnet"]