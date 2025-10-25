# -*- coding: gbk -*-
import torch
import os
import numpy as np
from Stage1_Model import MessageExtractor as Decoder
from torchvision.utils import save_image
from einops import rearrange
import torch.distributed as dist
from PIL import Image, ImageOps
from torchvision import transforms

def load_model(model_path):
    model = torch.load(model_path, map_location=torch.device('cuda:0'))    # or cuda, anyway we like #
    model.eval()
    return model

def enc():
    model_path = 'models/encoder.pth'
    encoder = load_model(model_path)
    
    # generate random bit string with length 30 for testing #
    bit_length = 30
    bit_array = np.random.randint(0, 2, size=bit_length)
    
    secret_input = torch.tensor(bit_array, dtype=torch.float32).cuda()
    
    secret_input = secret_input.unsqueeze(0)
    
    
    with torch.no_grad():
        secret_pattern = encoder(secret_input)
    
    return secret_pattern, secret_input


def dec(pattern):
    model_path = 'models/decoder.pth'

    decoder = Decoder()
    new_state = {}
    decoder_dict = torch.load(model_path)
    
    for key, value in decoder_dict.items():
        new_state[key.replace('module.', '')] = value
        
    decoder.load_state_dict(new_state)
    decoder = decoder.cuda()
    decoder.eval()

    with torch.no_grad():
        secret_output = decoder(pattern)
    
    return torch.round(secret_output)



def embed():
    model_path = 'saved_models/embedder.pth'
    embedder = load_model(model_path)

    # Read I_ori in #
    to_tensor = transforms.ToTensor()
    img_ori_path = 'I_ori.png'


    I_ori = Image.open(img_ori_path).convert('RGB')
    I_ori = ImageOps.fit(I_ori, (200, 200))
    I_ori = to_tensor(I_ori).cuda()
    I_ori = I_ori.unsqueeze(0)

    secret_pattern, secret_input = enc()

    I_w = embedder((secret_pattern, I_ori)) + I_ori
    I_w = torch.clamp(I_w, 0, 1)
    save_image(I_w, 'I_w.png')

    return secret_input


def partial_theft():
    I_w = np.array(Image.open('I_w.png'))
    I_bg = np.array(Image.open('I_bg.png'))
    I_mask = np.array(Image.open('I_mask.png'))
    
    if len(I_mask.shape) == 3:
        I_mask = I_mask[:, :, 0] if I_mask.shape[2] >= 1 else I_mask.mean(axis=2)
    
    mask_input = I_mask.astype(np.float32) / 255.0
    
    if len(I_w.shape) == 3:
        mask_input = np.expand_dims(mask_input, axis=2)
    
    I_compo = mask_input * I_w + (1 - mask_input) * I_bg

    I_compo = I_compo.astype(np.uint8)
    Image.fromarray(I_compo).save('I_compo.png')
    
    return I_compo



def extract():
    model_path = 'saved_models/extractor.pth'
    extractor = load_model(model_path)

    # Read I_compo (or I_w) in #
    to_tensor = transforms.ToTensor()
    img_path = 'I_compo.png'


    I_todecode = Image.open(img_path).convert('RGB')
    I_todecode = to_tensor(I_todecode).cuda()
    I_todecode = I_todecode.unsqueeze(0)

    extracted_pattern = extractor(I_todecode)
    decoded_secret = dec(extracted_pattern)

    return decoded_secret



if __name__ == "__main__":
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '19889'
    dist.init_process_group(backend='nccl', rank=0, world_size = 1)
    print(torch.__version__)
    
    secret_input = embed()
    partial_theft()
    secret_output = extract()