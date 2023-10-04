import argparse


def get_args():
    args = argparse.ArgumentParser(description="Train")
    args.add_argument('data_directory', action="store")

    args.add_argument('--save_dir',action="store",default="./checkpoint.pth",dest='save_dir',help='save checkpoint file')
    args.add_argument('--arch',action="store",default="vgg16",dest='arch',type=str)

    args.add_argument('--gpu',action="store_true",help='Use GPU')


    args.add_argument('--epochs',action="store",dest="epochs",default=5,type=int,help='Epochs')

    args.parse_args()
    return args


