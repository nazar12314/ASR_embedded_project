import torch
import yaml
import argparse

from onnxruntime.quantization import quantize_dynamic, QuantType
from transformers import Wav2Vec2ForCTC


def load_config(config_path):
    with open(config_path, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def convert_to_onnx(model_path, onnx_config):
    print(f"Converting {model_path} to ONNX")

    onnx_model_name = onnx_config.pop("model_path")

    model = Wav2Vec2ForCTC.from_pretrained(model_path)
    x = torch.randn(1, 32000, requires_grad=True)

    torch.onnx.export(model, x, onnx_model_name, **onnx_config)


def quantize_onnx(onnx_model_path, quantized_model_path):
    print(f"Starting quantization of {onnx_model_path}")

    quantize_dynamic(
        onnx_model_path,
        quantized_model_path,
        weight_type=QuantType.QUInt8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--quantize", type=bool, default=False)
    args = parser.parse_args()

    config = load_config(args.config)
    model_path = config["model"]["checkpoint"]
    quantization_model_path = config["quantization_parameters"]["model_path"]
    onnx_model_path = config["onnx_parameters"]["model_path"]
    onnx_config = config["onnx_parameters"]

    convert_to_onnx(model_path, onnx_config)

    if args.quantize:
        quantize_onnx(onnx_model_path, quantization_model_path)


if __name__ == "__main__":
    main()
