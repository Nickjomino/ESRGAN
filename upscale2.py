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
    if 'params' in state_dict and 'body.0.weight' in state_dict['params'].keys():
        print("Usando SRVGGNetCompact (Real-ESRGAN v2 Compact)")
        return SRVGGNetCompact(state_dict)
    # SPSR (ESRGAN com camadas extras)
    if 'f_HR_conv1.0.weight' in state_dict.keys():
        print("Usando SPSRNet (SPSR)")
        return SPSRNet(state_dict)
    # ESRGAN clássico / Real-ESRGAN v1 (RRDBNet)
    if 'model.0.weight' in state_dict.keys():
        print("Usando RRDBNet (ESRGAN clássico)")
        return RRDBNet(state_dict)
    raise ValueError("Formato de modelo não reconhecido.")


def process_image(model, device, img: Image.Image):
    """
    Executa inference no PIL Image e retorna PIL Image de saída.
    """
    arr = np.array(img).astype(np.float32) / 255.0
    # garante 3 canais
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=2)
    # converte HWC->CHW e cria batch
    tensor = torch.from_numpy(arr.transpose(2,0,1)).unsqueeze(0).to(device).half()
    with torch.no_grad():
        out = model(tensor).float().cpu().clamp_(0,1).numpy()[0]
    out = (out.transpose(1,2,0) * 255.0).round().astype(np.uint8)
    return Image.fromarray(out)


def main():
    parser = argparse.ArgumentParser(description="CLI para upscaling usando ESRGAN-Bot core (sem GUI)")
    parser.add_argument('--model', '-m', required=True,
                        help="Model string ou alias (ex: 4xPSNR, realesr-general-x4v3)")
    parser.add_argument('--input', '-i', required=True,
                        help="Pasta com imagens de entrada")
    parser.add_argument('--output', '-o', required=True,
                        help="Pasta para salvar imagens resultantes")
    args = parser.parse_args()

    # dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # parse_models retorna lista de cadeias (interpolations/chains)
    jobs = helpers.parse_models(args.model, helpers.aliases, helpers.fuzzymodels)
    if not jobs:
        print(f"Nenhum modelo encontrado para alias '{args.model}'")
        return

    os.makedirs(args.output, exist_ok=True)

    # processa cada imagem
    for fname in os.listdir(args.input):
        if not fname.lower().endswith(('.png','.jpg','.jpeg','.bmp','.webp')):
            continue
        img = Image.open(os.path.join(args.input, fname)).convert('RGB')
        result = img
        # aplica cada estágio da cadeia
        for stage in jobs[0]:
            # cada stage: {'model_name': ..., 'state_dict': ...}
            state_dict = stage['state_dict']
            net = build_model(state_dict).to(device).eval()
            result = process_image(net, device, result)
        # salva
        out_path = os.path.join(args.output, fname)
        result.save(out_path)
        print(f"Salvo: {out_path}")


if __name__ == '__main__':
    main()
