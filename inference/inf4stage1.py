# -*- coding: gbk -*-
import torch
import numpy as np
from torchvision.utils import save_image
from einops import rearrange

def load_model(model_path):
    model = torch.load(model_path, map_location=torch.device('cpu'))    # or cuda, anyway you like #
    model.eval()
    return model

def enc():
    model_path = 'models/encoder.pth'
    encoder = load_model(model_path)
    
    # generate random bit string with length 30 for testing #
    bit_length = 30
    bit_array = np.random.randint(0, 2, size=bit_length)
    
    secret_input = torch.tensor(bit_array, dtype=torch.float32)
    
    secret_input = secret_input.unsqueeze(0)    # b = 1 for now #
    print(f"Input Shape: {secret_input.shape}")
    
    secret_re = rearrange(secret_input, 'b h -> b 1 h 1')
    
    with torch.no_grad():
        secret_pattern = encoder(secret_re)
    
    return secret_pattern
    # save_image(secret_pattern + 0.5, 'test.png')

def dec(pattern):
    model_path = 'models/decoder.pth'
    decoder = load_model(model_path)

    with torch.no_grad():
        secret_output = decoder(pattern)
    
    return torch.round(secret_output)

if __name__ == "__main__":
    pattern = enc()
    output = dec(pattern)
    print(output)
    