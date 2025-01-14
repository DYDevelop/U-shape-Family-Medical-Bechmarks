import torch
import numpy as np
import os
import argparse
from torch.utils.data import DataLoader
from src.dataloader.dataset import MedicalDataSets
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
import src.utils.losses as losses
from src.utils.metrics import iou_score, multiclass_metrics
from torchvision.utils import save_image
# Assuming model imports based on your provided training script
from src.network.conv_based.CMUNet import CMUNet
from src.network.conv_based.U_Net import U_Net
from src.network.conv_based.AttU_Net import AttU_Net
from src.network.conv_based.UNeXt import UNext
from src.network.conv_based.UNetplus import ResNet34UnetPlus
from src.network.conv_based.UNet3plus import UNet3plus
from src.network.conv_based.CMUNeXt import cmunext
from src.network.transfomer_based.transformer_based_network import get_transformer_based_model
import cv2
import shutil

def load_model(model_path, args, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    # Model selection based on argument
    if args.model == "CMUNet":
        model = CMUNet(output_ch=args.num_classes)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model)
            # model.cuda()
    elif args.model == "CMUNeXt":
        model = cmunext(num_classes=args.num_classes)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model)
            # model.cuda()
    elif args.model == "U_Net":
        model = U_Net(output_ch=args.num_classes)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model)
            # model.cuda()
    elif args.model == "AttU_Net":
        model = AttU_Net(output_ch=args.num_classes)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model)
            # model.cuda()
    elif args.model == "UNext":
        model = UNext(output_ch=args.num_classes)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model)
            # model.cuda()
    elif args.model == "UNetplus":
        model = ResNet34UnetPlus(num_class=args.num_classes)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model)
            # model.cuda()
    elif args.model == "UNet3plus":
        model = UNet3plus(n_classes=args.num_classes)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model)
            # model.cuda()
    else:
        # Adjust accordingly for transformer-based models
        model = get_transformer_based_model(model_name=args.model, img_size=args.img_size, num_classes=args.num_classes, in_ch=3)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def get_val_transform(img_size):
    return Compose([
        Resize(img_size, img_size),
        Normalize(),
        # ToTensorV2(),
    ])

def validate(model, val_loader, criterion, device, save_dir="validation_results", num_classes=1):
    model.eval()
    val_loss = 0.0
    os.makedirs(save_dir, exist_ok=True)
    with open('/mnt/g/Prostate/data/axi/axi_val.txt', 'r') as f:
        sample_list = f.readlines()
    sample_list = [item.replace("\n", "") for item in sample_list]
    fold_metrics  = [{'iou':[], 'dsc':[], 'sensitivity':[], 'specificity':[], 'precision':[], 'accuracy':[], 'f1_score':[]} for _ in range(num_classes)]
    with torch.no_grad():
        for i_batch, sampled_batch in enumerate(val_loader):
            img_batch, label_batch, batch_idx = sampled_batch['image'], sampled_batch['label'], sampled_batch['idx']
            img_batch, label_batch = img_batch.to(device), label_batch.to(device)
            outputs = model(img_batch)
            loss = criterion(outputs, label_batch)

            val_loss += loss.item()

            iter_eval = multiclass_metrics(outputs, label_batch)
            if i_batch % 10 == 0:
                outputs = torch.squeeze(torch.sigmoid(outputs[:, 0, ...]))
                outputs = outputs.cpu().numpy()
                outputs[outputs > 0.5] = 1
                outputs[outputs <= 0.5] = 0
                output_images = outputs.astype(np.uint8) * 255
                
                for idx, (im_idx, msk) in enumerate(zip(batch_idx, output_images)):
                    img_np = cv2.resize(cv2.imread('/mnt/g/Prostate/data/axi/images/'+sample_list[im_idx]+'.png'), (args.img_size, args.img_size))
                    msk = np.stack([msk]*3, axis=-1)
                    overlay = cv2.addWeighted(img_np, 0.5, msk, 0.5, 0)
                    save_path = os.path.join(save_dir, f"batch_{i_batch}_img_{idx}.png")
                    cv2.imwrite(save_path, overlay)
                    # save_image(overlay, save_path)

            for classes in range(num_classes):
                for key in fold_metrics[classes].keys():fold_metrics[classes][key].extend(iter_eval[classes][key])

    for classes in range(num_classes):
        for key in fold_metrics[classes].keys(): 
            fold_metrics[classes][key] = sum(fold_metrics[classes][key]) / len(fold_metrics[classes][key]) if fold_metrics[classes][key] else 0.0
    for classes in range(num_classes):
        print(f'-- Currnet fold Class {classes} evaluation metrics --')
        for key, values in fold_metrics[classes].items(): print(f'{key}: {values:.3f}')

    return fold_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validation script for medical image segmentation")
    parser.add_argument('--model', type=str, default="U_Net", help='model type')
    parser.add_argument('--model_path', type=str, default="./checkpoint/U_Net_model.pth", help='Path to the trained model')
    parser.add_argument('--base_dir', type=str, default="./data/test", help='base directory of dataset')
    parser.add_argument('--val_file_dir', type=str, default="test_val.txt", help='validation file directory')
    parser.add_argument('--img_size', type=int, default=256, help='image size')
    parser.add_argument('--num_classes', type=int, default=1, help='number of classes')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--k_fold', type=int, default=5, help='number of folds')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total_metrics  = [{'iou':[], 'dsc':[], 'sensitivity':[], 'specificity':[], 'precision':[], 'accuracy':[], 'f1_score':[]} for _ in range(args.num_classes)]

    if os.path.exists("validation_results"): shutil.rmtree("validation_results")

    for fold in range(args.k_fold):
        model = load_model(args.model_path+args.model+f'_{fold}_model.pth', args, device)

        val_transform = get_val_transform(args.img_size)

        db_val = MedicalDataSets(base_dir=args.base_dir, split="val", transform=val_transform, val_file_dir=args.val_file_dir, n_classes=args.num_classes)
        val_loader = DataLoader(db_val, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        criterion = losses.__dict__['BCEDiceLoss']().to(device)
        fold_metrics = validate(model, val_loader, criterion, device, num_classes=args.num_classes)

        for classes in range(args.num_classes):
            for key in total_metrics[classes].keys():total_metrics[classes][key].append(fold_metrics[classes][key])

    for classes in range(args.num_classes):
        print(f'-- Final Class {classes} evaluation metrics --')
        for key in total_metrics[classes].keys(): 
            print(f"{key} : {np.mean(total_metrics[classes][key]):.3f} ± {np.std(total_metrics[classes][key]):.3f}")

# python infer.py --model U_Net --model_path checkpoint/ --base_dir ../data/axi --val_file_dir axi_val.txt --img_size 256 --num_classes 2