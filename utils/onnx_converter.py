import torch
import argparse

from onnxruntime.quantization import quantize_dynamic, QuantType
from transformers import Wav2Vec2ForCTC
from typing import Dict

from utils.logger_utils import get_logger
from utils.config_utils import load_config


logger = get_logger(__name__)


def convert_to_onnx(config: Dict):
    """
    Converts PyTorch model to ONNX format

    :param config: dictionary with configuration
    """
    checkpoint_path = config["checkpoint"]
    onnx_config = config["onnx_parameters"]
    onnx_path = config["output"]["onnx"]

    logger.info(f"Converting {checkpoint_path} to ONNX")

    model = Wav2Vec2ForCTC.from_pretrained(checkpoint_path)
    x = torch.randn(1, 32000, requires_grad=True)

    torch.onnx.export(model, x, onnx_path, **onnx_config)


def quantize_onnx(config: Dict):
    """
    Quantizes ONNX model

    :param config: dictionary with configuration
    """
    onnx_path = config["output"]["onnx"]
    quantized_path = config["output"]["quantized"]

    logger.info(f"Starting quantization of {onnx_path}")

    quantize_dynamic(
        onnx_path,
        quantized_path,
        weight_type=QuantType.QUInt8
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--quantize", type=bool, default=False)
    args = parser.parse_args()

    quant_config = load_config(args.config)
    convert_to_onnx(quant_config)

    if args.quantize:
        quantize_onnx(quant_config)
