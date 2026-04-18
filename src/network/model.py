from src.network.conv_based.CMUNet import CMUNet
from src.network.conv_based.U_Net import U_Net
from src.network.conv_based.AttU_Net import AttU_Net
from src.network.conv_based.UNeXt import UNext
from src.network.conv_based.UNetplus import ResNet34UnetPlus
from src.network.conv_based.UNet3plus import UNet3plus
from src.network.conv_based.CMUNeXt import cmunext

from src.network.transfomer_based.transformer_based_network import get_transformer_based_model
import segmentation_models_pytorch as smp

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

def get_model_smp(args):
    if args.model_smp == "MAnet":
        model = smp.MAnet(
                        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                        classes=args.num_classes,                      # model output channels (number of classes in your dataset)
                    )
    elif args.model_smp == "Linknet":
        model = smp.Linknet(
                        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                        classes=args.num_classes,                      # model output channels (number of classes in your dataset)
                    )
    elif args.model_smp == "Unet":
        model = smp.Unet(
                        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                        classes=args.num_classes,                      # model output channels (number of classes in your dataset)
                    )
    elif args.model_smp == "FPN":
        model = smp.FPN(
                        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                        classes=args.num_classes,                      # model output channels (number of classes in your dataset)
                    )
    elif args.model_smp == "PSPNet":
        model = smp.PSPNet(
                        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                        classes=args.num_classes,                      # model output channels (number of classes in your dataset)
                    )
    elif args.model_smp == "DeepLabV3":
        model = smp.DeepLabV3(
                        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                        classes=args.num_classes,                      # model output channels (number of classes in your dataset)
                    )
    elif args.model_smp == "DeepLabV3Plus":
        model = smp.DeepLabV3Plus(
                        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                        classes=args.num_classes,                      # model output channels (number of classes in your dataset)
                    )
    elif args.model_smp == "PAN":
        model = smp.PAN(
                        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                        classes=args.num_classes,                      # model output channels (number of classes in your dataset)
                    )
    elif args.model_smp == "UnetPlusPlus":
        model = smp.UnetPlusPlus(
                        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                        classes=args.num_classes,                      # model output channels (number of classes in your dataset)
                    )
    elif args.model_smp == "UPerNet":
        model = smp.UPerNet(
                        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                        classes=args.num_classes,                      # model output channels (number of classes in your dataset)
                    )
    elif args.model_smp == "Segformer":
        model = smp.Segformer(
                        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                        classes=args.num_classes,                      # model output channels (number of classes in your dataset)
                    )
    elif args.model_smp == "DPT":
        model = smp.DPT(
                        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                        classes=args.num_classes,                      # model output channels (number of classes in your dataset)
                    )
    else:
        print("No model found!")
        exit()
        
    return model.cuda()