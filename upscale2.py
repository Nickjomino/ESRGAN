import os
import argparse
import torch
import numpy as np
from PIL import Image

# importa funções de alias e parsing
import helpers
from utils.architecture.RRDB import RRDBNet
from utils.architecture.SRVGG import SRVGGNetCompact
from utils.architecture.SPSR import SPSRNet

def build_model(state_dict):
    """
    Constrói a rede correta a partir do state_dict carregado.
    """
    # Real-ESRGAN v2 Compact (SRVGGNetCompact)
    if any(k.startswith('body.0') for k in state_dict.keys()):
        return SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')

    # SPSRNet (SwinIR-like)
    if 'first_part.0.weight' in state_dict:
        return SPSRNet()

    # RRDBNet padrão
    return RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23)

def load_image(path):
    return Image.open(path).convert("RGB")

def save_image(img_tensor, path):
    img = img_tensor.squeeze().float().cpu().clamp_(0, 1).numpy()
    img = (img * 255.0).round().astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))
    Image.fromarray(img).save(path)

def upscale_image(model, device, image_path, output_path):
    image = load_image(image_path)
    img_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.
    img_tensor = img_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
    save_image(output, output_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Alias ou caminho de .pth')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--half', action='store_true')
    args = parser.parse_args()

    model_paths = helpers.parse_models(args.model)
    state_dict = helpers.load_weights_chain(model_paths)

    model = build_model(state_dict)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(args.device)
    if args.half:
        model.half()

    os.makedirs(args.output, exist_ok=True)
    images = [f for f in os.listdir(args.input) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    for img_name in images:
        input_path = os.path.join(args.input, img_name)
        output_path = os.path.join(args.output, img_name)
        upscale_image(model, args.device, input_path, output_path)

if __name__ == '__main__':
    main()
