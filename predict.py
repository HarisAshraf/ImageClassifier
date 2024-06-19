#!/usr/bin/python3
import numpy as np
import argparse
import torch
from torch import optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.autograd import Variable
from collections import OrderedDict
import matplotlib.pyplot as plt
import json
from PIL import Image


parser = argparse.ArgumentParser(
    description = 'Parser for predict function'
)

parser.add_argument('image_path', action = 'store')
parser.add_argument('checkpoint', action = 'store')
parser.add_argument('--category_names', action = 'store', default='cat_to_name.json')
parser.add_argument('--top_k', action = 'store', default=5)
parser.add_argument('--gpu', action = 'store', default='gpu')
parser.add_argument('--arch', action = 'store', default='vgg15')

args = parser.parse_args()

# Arguments are global, although only used in main
image_path = args.image_path
checkpoint = args.checkpoint
category_name = args.category_names 
top_k = int(args.top_k)
gpu = args.gpu

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

image_path ='flowers/test/5/image_05159.jpg'
checkpoint ='checkpoint.pth'
category_name = cat_to_name

# Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(checkpoint):

   checkpoint = torch.load(checkpoint)

   if args.arch=='densenet121':
      model = models.densenet121(pretrained=True)   
   elif args.arch=='vgg13':
      model = models.vgg13(pretrained=True)
   else: 
      model = models.vgg16(pretrained=True)

   model.classifier = checkpoint['model_classifier']
   model.load_state_dict(checkpoint['model_state_dict'])
   model.class_to_idx = checkpoint ['train_data_class_to_idx']
              
   optimizer = optim.Adam(model.classifier.parameters(), lr=0.002)
   optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 

   return model, optimizer             
              

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Process a PIL image for use in a PyTorch model
    
    # Define a transform to convert PIL image to a Torch tensor
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])
    # Convert the PIL image to Torch tensor
    img_tensor = transform(image)   
    
    return img_tensor


def predict(image_path, model, gpu, top_k):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    #Open Image
    jpeg_image=Image.open(image_path)

    #Process and convert into a Tensor
    image_tensor = process_image(jpeg_image)
    

    image = image_tensor.numpy()    
    image = torch.from_numpy(np.array([image]))
    
    # Implement the code to predict the class from an image file
    # Use GPU if it's available
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device("cuda" if gpu else "cpu")
    
    model=model.to(device)
    image=image.to(device)
    
    model.eval
    
    # Calculate the class probabilities for image
    with torch.no_grad():
        output = model.forward(image)
        
    prob = torch.exp(output)
    
    return prob.topk(top_k)

def main():


    model, optimizer = load_checkpoint(checkpoint)

    #Open Image
    jpeg_image=Image.open(image_path)

    #Process and convert into a Tensor
    image_tensor = process_image(jpeg_image)
    
    image = image_tensor.numpy()    
    image = torch.from_numpy(np.array([image]))

    image=Image.open(image_path)

    tensor_image = process_image(image)

    predictions = predict(image_path, model, gpu =='gpu', top_k)


    # Move predictions back to CPU
    probs=predictions[0][0].cpu()
    cat=predictions[1][0].cpu()

    # Get idex for title and convert from Tensor
    title_index = cat[0].numpy()
    flower_cats = cat.numpy().tolist()


    ax1 = imshow(tensor_image, ax = plt)
    ax1.axis('off')
    ax1.title(cat_to_name[str(title_index)])

    x_axis = np.array(probs)
    y_axis=[]
    for i in flower_cats:
        y_axis.append(cat_to_name[str(i)])

    fig,ax2 = plt.subplots(figsize=(top_k,top_k))


    y_pos = np.arange(top_k)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(y_axis)
    ax2.set_xlabel('Probability')
    ax2.invert_yaxis()
    ax2.barh(y_axis, x_axis)

    plt.show()

if __name__ == "__main__":
    main()
