import os
import argparse
import torch
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.archs.srvgg_arch import SRVGGNetCompact
from realesrgan import RealESRGANer

def load_model(model_path, scale):
    print(f"Carregando modelo: {model_path}")
    state_dict = torch.load(model_path, map_location='cpu')

    if "params" in state_dict and "body.0.weight" in state_dict["params"]:
        print("Detectado modelo Real-ESRGAN v2 Compact (SRVGGNetCompact)")
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64,
                                num_conv=32, upscale=scale, act_type='prelu')
        model.load_state_dict(state_dict["params"], strict=True)
    elif "model.0.weight" in state_dict:
        print("Detectado modelo ESRGAN clássico (RRDBNet)")
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=scale)
        model.load_state_dict(state_dict, strict=True)
    else:
        raise ValueError("Modelo não reconhecido. Estrutura de state_dict não compatível.")

    return model

def upscale(model, device, input_dir, output_dir, scale):
    upsampler = RealESRGANer(
        scale=scale,
        model_path=None,  # já carregamos o modelo manualmente
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False
    )

    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
            input_path = os.path.join(input_dir, file)
            output_path = os.path.join(output_dir, file)

            img = Image.open(input_path).convert("RGB")
            try:
                output, _ = upsampler.enhance(img, outscale=scale)
                output.save(output_path)
                print(f"Salvo: {output_path}")
            except Exception as e:
                print(f"Erro ao processar {file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Caminho para o modelo .pth")
    parser.add_argument("--scale", type=int, default=2, help="Fator de upscale")
    parser.add_argument("--input", required=True, help="Pasta com imagens de entrada")
    parser.add_argument("--output", required=True, help="Pasta para salvar imagens processadas")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, args.scale).to(device)
    model.eval()

    upscale(model, device, args.input, args.output, args.scale)
