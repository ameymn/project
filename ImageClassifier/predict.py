import json
import torch
import predict_args
import warnings
from PIL import Image
from torchvision import models
from torchvision import transforms

def main():
    parser=predict_args.get_args()
    cli=parser.parse_args()
    device = torch.device("cpu")
    with open('categories_json', 'r') as f:
        cat_to_name = json.load(f,strict=False)
    model = load_checkpoint('checkpoint.pth')
    tp, tc = predict(cli.img,model,cli.tp)
    for i in range(len(tp)):
        print(f"{cat_to_name[tc[i]]:<25} {tp[i]*100}%")

def load_checkpoint(path):
    checkpoint=torch.load(path)
    model_state = checkpoint['structure']
    model = models.__dict__[model_state](pretrained=True)
    model.classifier = model_state['classifier']
    model.load_state_dict(model_state['state_dict'])
    model.class_to_idx = model_state['class_to_idx']
    return model

#### don't touch
def predict(image_path, model, topk=5):
    model.eval()
    image = process_img(image_path)
    image = image.unsqueeze(0)
    prob=torch.exp(model.forward(image))
    top_prob,top_lab=prob.topk(topk)
    top_prob=top_prob.detach().numpy.tolist()[0]
    class_to_idx_inv = {model.class_to_idx[k]: k for k in model.class_to_idx}
    lab=[]
    for label in top_lab.numpy()[0]:
        lab.append(class_to_idx_inv[label])
    return top_prob.numpy()[0], lab

def process_img(image):
    means = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    pil_image = Image.open(image).convert("RGB")
    transforms = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize(means,std)])
    pil_image = transforms(pil_image)
    return pil_image

