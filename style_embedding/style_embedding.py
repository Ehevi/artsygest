import os
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from PIL import Image
from CaffeLoader import loadCaffemodel, ModelParallel

from tqdm import tqdm

OUT_PATH = Path("/media/pawel/DATA/iwisum/results/embeddings/")

import argparse
parser = argparse.ArgumentParser()
# Basic options
parser.add_argument("-style_path", help="Style target image", default='examples/inputs/seated-nude.jpg')
parser.add_argument("-image_size", help="Maximum height / width of generated image", type=int, default=512)
parser.add_argument("-gpu", help="Zero-indexed ID of the GPU to use; for CPU mode set -gpu = c", default=0)

# Optimization options
parser.add_argument("-normalize_weights", action='store_true')

# Output options
parser.add_argument("-output_tensor", default='style_embeddings.pt')

# Other options
parser.add_argument("-style_scale", type=float, default=1.0)
parser.add_argument("-pooling", choices=['avg', 'max'], default='max')
parser.add_argument("-model_file", type=str, default='models/vgg19-d01eb7cb.pth')
parser.add_argument("-disable_check", action='store_true')
parser.add_argument("-backend", choices=['nn', 'cudnn', 'mkl', 'mkldnn', 'openmp', 'mkl,cudnn', 'cudnn,mkl'], default='nn')
parser.add_argument("-cudnn_autotune", action='store_true')
parser.add_argument("-style_layers", help="layers for style", default='relu1_1,relu2_1,relu3_1,relu4_1,relu5_1')
parser.add_argument("-multidevice_strategy", default='4,7,29')
parser.add_argument("-verbose", action=argparse.BooleanOptionalAction)

params = parser.parse_args()


Image.MAX_IMAGE_PIXELS = 1000000000 # Support gigapixel images


def main():
    dtype, multidevice= setup_gpu()
    cnn, layerList = loadCaffemodel(params.model_file, params.pooling, params.gpu, params.disable_check, params.verbose)
    
    style_image_input = params.style_path.split(',')
    dataset_name = style_image_input[0].split('/')[-1]

    style_image_list, ext = [], [".jpg", ".jpeg", ".png", ".tiff"]
    print("Loading images...")
    for image in tqdm(style_image_input):
        if os.path.isdir(image):
            images = (image + "/" + file for file in os.listdir(image)
            if os.path.splitext(file)[1].lower() in ext)
            style_image_list.extend(images)
        else:
            style_image_list.append(image)
    print(f"Number of loaded images: {len(style_image_list)}")
    with open(f'style_images_{dataset_name}.txt', 'w') as f:
        for item in style_image_list:
            f.write("%s\n" % item)

    
    style_layers = params.style_layers.split(',')
    
    print(f"Capturing styles...")
    split_image_list = [style_image_list[i:i + 100] for i in range(0, len(style_image_list), 100)]
    for i, split_image in tqdm(enumerate(split_image_list)):
        perform_style_extraction(dtype, multidevice, cnn, layerList, dataset_name, split_image, style_layers, i)
    print(f"Done capturing styles.")

def perform_style_extraction(dtype, multidevice, cnn, layerList, dataset_name, style_image_list, style_layers, number):
    style_embeddings_per_layer = [[] for _ in range(len(style_layers))]
    for img_name in tqdm(style_image_list):
        img_caffe = prepare_img_caffe(dtype, img_name)
        if img_caffe is None: continue

        style_losses, net = set_up_network(multidevice, cnn, layerList, style_layers)
        if params.verbose: print("Capturing style target " + img_name)
        
        for style_layer in style_losses:
            style_layer.mode = 'capture'

        net(img_caffe)
        flat_gram_matrices = [style_loss.target.flatten() for style_loss in style_losses]
        if params.normalize_weights:
            flat_gram_matrices= [gram_matrix / gram_matrix.shape[0] for gram_matrix in flat_gram_matrices]
        for layer_embedding, gram_matrix in zip(style_embeddings_per_layer, flat_gram_matrices):
            layer_embedding.append(gram_matrix)

    for embeddings, layer_name in zip(style_embeddings_per_layer, style_layers):
        style_embeddings_tensor = torch.stack(embeddings)
        
        if params.verbose: print(f"Style embeddings tensor size for layer {layer_name}: {style_embeddings_tensor.size()}")
        
        out_name = f"{OUT_PATH}/{dataset_name}/style_embeddings_{dataset_name}_{layer_name}_{number}.pt"
        os.makedirs(f"{OUT_PATH}/{dataset_name}", exist_ok=True)

        print(f"Saving style embeddings tensor for layer {layer_name} to `{out_name}`")
        torch.save(style_embeddings_tensor, out_name)
    


def prepare_img_caffe(dtype, image):
    style_size = int(params.image_size * params.style_scale)
    try:
        return preprocess(image, style_size).type(dtype)
    except:
        print("Error loading image %s:" % image)
        return None     
    
    
   
def set_up_network(multidevice, cnn, layerList, style_layers):
    cnn = copy.deepcopy(cnn)
    style_losses = []
    next_style_idx = 1
    net = nn.Sequential()
    c, r = 0, 0

    for i, layer in enumerate(list(cnn), 1):
        if next_style_idx <= len(style_layers):
            if isinstance(layer, nn.Conv2d):
                net.add_module(str(len(net)), layer)

                if layerList['C'][c] in style_layers:
                    if params.verbose: print("Setting up style layer " + str(i) + ": " + str(layerList['C'][c]))
                    loss_module = StyleLoss()
                    net.add_module(str(len(net)), loss_module)
                    style_losses.append(loss_module)
                c+=1

            if isinstance(layer, nn.ReLU):
                net.add_module(str(len(net)), layer)

                if layerList['R'][r] in style_layers:
                    if params.verbose: print("Setting up style layer " + str(i) + ": " + str(layerList['R'][r]))
                    loss_module = StyleLoss()
                    net.add_module(str(len(net)), loss_module)
                    style_losses.append(loss_module)
                    next_style_idx += 1
                r+=1

            if isinstance(layer, nn.MaxPool2d) or isinstance(layer, nn.AvgPool2d):
                net.add_module(str(len(net)), layer)

    if multidevice:
        net = setup_multi_device(net)
    return style_losses,net


def setup_gpu():
    def setup_cuda():
        if 'cudnn' in params.backend:
            torch.backends.cudnn.enabled = True
            if params.cudnn_autotune:
                torch.backends.cudnn.benchmark = True
        else:
            torch.backends.cudnn.enabled = False

    def setup_cpu():
        if 'mkl' in params.backend and 'mkldnn' not in params.backend:
            torch.backends.mkl.enabled = True
        elif 'mkldnn' in params.backend:
            raise ValueError("MKL-DNN is not supported yet.")
        elif 'openmp' in params.backend:
            torch.backends.openmp.enabled = True

    multidevice = False
    if "," in str(params.gpu):
        devices = params.gpu.split(',')
        multidevice = True

        if 'c' in str(devices[0]).lower():
            setup_cuda(), setup_cpu()
        else:
            setup_cuda()
        dtype = torch.FloatTensor

    elif "c" not in str(params.gpu).lower():
        setup_cuda()
        dtype = torch.cuda.FloatTensor
    else:
        setup_cpu()
        dtype = torch.FloatTensor
    return dtype, multidevice


def setup_multi_device(net):
    assert len(params.gpu.split(',')) - 1 == len(params.multidevice_strategy.split(',')), \
      "The number of -multidevice_strategy layer indices minus 1, must be equal to the number of -gpu devices."

    new_net = ModelParallel(net, params.gpu, params.multidevice_strategy)
    return new_net


# Preprocess an image before passing it to a model.
# We need to rescale from [0, 1] to [0, 255], convert from RGB to BGR,
# and subtract the mean pixel.
def preprocess(image_name, image_size):
    
    with Image.open(image_name) as image:
        image = image.convert('RGB')
    if type(image_size) is not tuple:
        image_size = tuple([int((float(image_size) / max(image.size))*x) for x in (image.height, image.width)])
    Loader = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
    rgb2bgr = transforms.Compose([transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])])])
    Normalize = transforms.Compose([transforms.Normalize(mean=[103.939, 116.779, 123.68], std=[1,1,1])])
    tensor = Normalize(rgb2bgr(Loader(image) * 255)).unsqueeze(0)
    return tensor


#  Undo the above preprocessing.
def deprocess(output_tensor):
    Normalize = transforms.Compose([transforms.Normalize(mean=[-103.939, -116.779, -123.68], std=[1,1,1])])
    bgr2rgb = transforms.Compose([transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])])])
    output_tensor = bgr2rgb(Normalize(output_tensor.squeeze(0).cpu())) / 255
    output_tensor.clamp_(0, 1)
    Image2PIL = transforms.ToPILImage()
    image = Image2PIL(output_tensor.cpu())
    return image


# Scale gradients in the backward pass
class ScaleGradients(torch.autograd.Function):
    @staticmethod
    def forward(self, input_tensor, strength):
        self.strength = strength
        return input_tensor

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        grad_input = grad_input / (torch.norm(grad_input, keepdim=True) + 1e-8)
        return grad_input * self.strength * self.strength, None


class GramMatrix(nn.Module):

    def forward(self, input):
        _B, C, H, W = input.size()
        x_flat = input.view(C, H * W)
        return torch.mm(x_flat, x_flat.t())


# Define an nn Module to compute style loss
class StyleLoss(nn.Module):

    def __init__(self):
        super(StyleLoss, self).__init__()
        self.target = torch.Tensor()
        self.gram = GramMatrix()
        self.mode = 'None'
        self.blend_weight = None

    def forward(self, input):
        self.G = self.gram(input)
        self.G = self.G.div(input.nelement())
        if self.mode == 'capture':
            if self.blend_weight == None:
                self.target = self.G.detach()
            elif self.target.nelement() == 0:
                self.target = self.G.detach().mul(self.blend_weight)
            else:
                self.target = self.target.add(self.blend_weight, self.G.detach())
        return input


if __name__ == "__main__":
    main()