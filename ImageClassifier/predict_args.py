import argparse

### done
def get_args():
    args=argparse.ArgumentParser(description='predict.py')
    args.add_argument('--gpu', dest="gpu", action="store", default="gpu")
    args.add_argument('img',action="store",help='image')
    args.add_argument('--category_names', dest= "category_names", action="store", default='cat_to_name.json')
    args.add_argument('checkpoint', nargs='*', action="store", type = str, default="/classifier.pth")
    args.add_argument('tp',action='store',default=5,type=int)
    return args

