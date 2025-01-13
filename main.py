import os
import argparse
import random
import numpy as np
import torch
import torch.optim as optim

from torch.utils.data import DataLoader
from src.dataloader.dataset import MedicalDataSets
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from albumentations import RandomRotate90, Resize, HorizontalFlip
import src.utils.losses as losses
from src.utils.util import AverageMeter
from src.utils.metrics import iou_score

from src.network.conv_based.CMUNet import CMUNet
from src.network.conv_based.U_Net import U_Net
from src.network.conv_based.AttU_Net import AttU_Net
from src.network.conv_based.UNeXt import UNext
from src.network.conv_based.UNetplus import ResNet34UnetPlus
from src.network.conv_based.UNet3plus import UNet3plus
from src.network.conv_based.CMUNeXt import cmunext

from src.network.transfomer_based.transformer_based_network import get_transformer_based_model

from infer import validate

def seed_torch(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="U_Net",
                    choices=["CMUNeXt", "CMUNet", "AttU_Net", "TransUnet", "R2U_Net", "U_Net",
                             "UNext", "UNetplus", "UNet3plus", "SwinUnet", "MedT", "TransUnet"], help='model')
parser.add_argument('--base_dir', type=str, default="./data/busi", help='dir')
parser.add_argument('--train_file_dir', type=str, default="busi_train.txt", help='dir')
parser.add_argument('--val_file_dir', type=str, default="busi_val.txt", help='dir')
parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--epoch', type=int, default=300, help='train epoch')
parser.add_argument('--patience', type=int, default=10, help='validation patience')
parser.add_argument('--k_fold', type=int, default=5, help='number of folds')
parser.add_argument('--img_size', type=int, default=256, help='img size of per batch')
parser.add_argument('--num_classes', type=int, default=1, help='seg num_classes')
parser.add_argument('--seed', type=int, default=41, help='random seed')
args = parser.parse_args()
seed_torch(args.seed)


def get_model(args):
    if args.model == "CMUNet":
        model = CMUNet(output_ch=args.num_classes).cuda()
    elif args.model == "CMUNeXt":
        model = cmunext(num_classes=args.num_classes).cuda()
    elif args.model == "U_Net":
        model = U_Net(output_ch=args.num_classes).cuda()
    elif args.model == "AttU_Net":
        model = AttU_Net(output_ch=args.num_classes).cuda()
    elif args.model == "UNext":
        model = UNext(output_ch=args.num_classes).cuda()
    elif args.model == "UNetplus":
        model = ResNet34UnetPlus(num_class=args.num_classes).cuda()
    elif args.model == "UNet3plus":
        model = UNet3plus(n_classes=args.num_classes).cuda()
    else:
        model = get_transformer_based_model(parser=parser, model_name=args.model, img_size=args.img_size,
                                            num_classes=args.num_classes, in_ch=3).cuda()
    return model

def getDataloader(args, fold):
    img_size = args.img_size
    if args.model == "SwinUnet":
        img_size = 224
    train_transform = Compose([
        # RandomRotate90(),
        # transforms.Flip(),
        HorizontalFlip(p=0.3),
        Resize(img_size, img_size),
        transforms.Normalize(),
    ])

    val_transform = Compose([
        Resize(img_size, img_size),
        transforms.Normalize(),
    ])
    db_train = MedicalDataSets(base_dir=args.base_dir, split="train",
                            transform=train_transform, train_file_dir=args.train_file_dir, val_file_dir=args.val_file_dir, n_classes=args.num_classes)
    db_test = MedicalDataSets(base_dir=args.base_dir, split="val", transform=val_transform,
                          train_file_dir=args.train_file_dir, val_file_dir=args.val_file_dir, n_classes=args.num_classes)
    total_size = len(db_train)
    fraction = 1/args.k_fold
    seg = int(total_size * fraction)

    trll = 0
    trlr = fold * seg
    vall = trlr
    valr = fold * seg + seg
    trrl = valr
    trrr = total_size

    train_left_indices = list(range(trll,trlr))
    train_right_indices = list(range(trrl,trrr))
    
    train_indices = train_left_indices + train_right_indices
    val_indices = list(range(vall,valr))
    
    train_set = torch.utils.data.dataset.Subset(db_train,train_indices)
    val_set = torch.utils.data.dataset.Subset(db_train,val_indices)

    print("train num:{}, val num:{}, test num:{}".format(len(train_set), len(val_set), len(db_test)))

    trainloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=False)
    valloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    testloader = DataLoader(db_test, batch_size=args.batch_size, shuffle=False, num_workers=4)

    return trainloader, valloader, testloader

def main(args):
    base_lr = args.base_lr
    total_metrics  = [{'iou':[], 'dsc':[], 'sensitivity':[], 'specificity':[], 'precision':[], 'accuracy':[], 'f1_score':[]} for _ in range(args.num_classes)]

    for fold in range(args.k_fold):
        print(f'------------- Fold {fold} Training Started -------------')

        trainloader, valloader, testloader = getDataloader(args=args, fold=fold)

        model = get_model(args)

        print("train file dir:{} val file dir:{}".format(args.train_file_dir, args.val_file_dir))

        optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
        criterion = losses.__dict__['BCEDiceLoss']().cuda()

        print("{} iterations per epoch".format(len(trainloader)))
        # best_iou = 0
        best_loss = float("inf")
        iter_num = 0
        patience = 0
        max_epoch = args.epoch

        max_iterations = len(trainloader) * max_epoch
        for epoch_num in range(max_epoch):
            model.train()
            avg_meters = {'loss': AverageMeter(),
                        'iou': AverageMeter(),
                        'dsc': AverageMeter(),
                        'val_loss': AverageMeter(),
                        'val_iou': AverageMeter(),
                        'val_dsc': AverageMeter(),
                        'val_SE': AverageMeter(),
                        'val_PC': AverageMeter(),
                        'val_F1': AverageMeter(),
                        'val_ACC': AverageMeter()}

            for i_batch, sampled_batch in enumerate(trainloader):

                img_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                img_batch, label_batch = img_batch.cuda(), label_batch.cuda()

                outputs = model(img_batch)
                
                loss = criterion(outputs, label_batch)
                iou, dice, _, _, _, _, _ = iou_score(outputs, label_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

                iter_num = iter_num + 1
                avg_meters['loss'].update(loss.item(), img_batch.size(0))
                avg_meters['iou'].update(iou, img_batch.size(0))
                avg_meters['dsc'].update(dice, img_batch.size(0))

            model.eval()
            with torch.no_grad():
                for i_batch, sampled_batch in enumerate(valloader):
                    img_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                    img_batch, label_batch = img_batch.cuda(), label_batch.cuda()
                    output = model(img_batch)
                    loss = criterion(output, label_batch)
                    iou, dice, SE, PC, F1, _, ACC = iou_score(output, label_batch)
                    avg_meters['val_loss'].update(loss.item(), img_batch.size(0))
                    avg_meters['val_iou'].update(iou, img_batch.size(0))
                    avg_meters['val_dsc'].update(dice, img_batch.size(0))
                    avg_meters['val_SE'].update(SE, img_batch.size(0))
                    avg_meters['val_PC'].update(PC, img_batch.size(0))
                    avg_meters['val_F1'].update(F1, img_batch.size(0))
                    avg_meters['val_ACC'].update(ACC, img_batch.size(0))

            print('epoch [%d/%d]  train_loss : %.4f, train_iou: %.4f, train_dsc: %.4f - val_loss %.4f - val_iou %.4f - val_dsc %.4f - val_SE %.4f - '
                'val_PC %.4f - val_F1 %.4f - val_ACC %.4f '
                % (epoch_num, max_epoch, avg_meters['loss'].avg, avg_meters['iou'].avg, avg_meters['dsc'].avg,
                avg_meters['val_loss'].avg, avg_meters['val_iou'].avg, avg_meters['val_dsc'].avg, avg_meters['val_SE'].avg,
                avg_meters['val_PC'].avg, avg_meters['val_F1'].avg, avg_meters['val_ACC'].avg))

            if avg_meters['val_loss'].avg < best_loss:
                if not os.path.isdir("./checkpoint"):
                    os.makedirs("./checkpoint")
                torch.save(model.state_dict(), 'checkpoint/{}_{}_model.pth'.format(args.model, fold))
                best_loss = avg_meters['val_loss'].avg
                print("=> saved best model")
                patience = 0
            else:
                patience += 1
                print(f'------------- Model did not improve for {patience} times -------------')

            if patience == args.patience:
                print("------------- Training Finished! -------------")
                break
        print(f"------------- Fold {fold} Evaluation on Testset -------------")
        model.load_state_dict(torch.load(f'checkpoint/{args.model}_{fold}_model.pth'))
        model.eval()
        fold_metrics = validate(model, testloader, criterion, "cuda", num_classes=args.num_classes)

        for classes in range(args.num_classes):
            for key in total_metrics[classes].keys():total_metrics[classes][key].append(fold_metrics[classes][key])

    print(f"------------------- All Folds Training Finished -------------------")
    for classes in range(args.num_classes):
        print(f'-- Final Class {classes} evaluation metrics --')
        for key in total_metrics[classes].keys(): 
            print(f"{key} : {np.mean(total_metrics[classes][key]):.3f} ± {np.std(total_metrics[classes][key]):.3f}")

if __name__ == "__main__":
    main(args)
