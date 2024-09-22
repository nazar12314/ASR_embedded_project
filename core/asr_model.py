import torch
import librosa
import argparse
import onnxruntime as rt
import numpy as np

from transformers import Wav2Vec2Processor
from utils.logger_utils import get_logger

from typing import Union


class QuantizedWave2Vec2ForCTC:
    def __init__(
            self,
            checkpoint_name: str = "facebook/wav2vec2-base-960h",
            model_path: str = "model.quant.onnx"
        ):

        self.model = self.load_model(model_path)
        self.processor = Wav2Vec2Processor.from_pretrained(checkpoint_name)

    def load_model(self, model_path):
        """
        Load ONNX model

        Args:
            model_path (str): path to ONNX model
        Returns:
            onnxruntime.InferenceSession: loaded model
        """
        options = rt.SessionOptions()
        options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        model = rt.InferenceSession(model_path, options)

        return model

    def predict(self, audio: Union[np.ndarray, str, torch.Tensor]):
        """
        Predict transcription from audio

        Args:
            audio_path (str): path to audio file

        Returns:
            str: transcription
            float: confidence score
        """
        if isinstance(audio, str):
            audio, _ = librosa.load(audio, sr=16000)
            audio = torch.tensor(audio, dtype=torch.float32)

            print(audio.shape)
        elif isinstance(audio, np.ndarray):
            audio = torch.tensor(audio, dtype=torch.float32)

        inputs = self.processor(audio, return_tensors="pt", sampling_rate=16000, padding=True)

        onnx_outputs = self.model.run(
            None,
            {self.model.get_inputs()[0].name: inputs.input_values.numpy()}
        )[0]

        predictions = np.argmax(onnx_outputs, axis=-1)
        transcription = self.processor.decode(predictions.squeeze().tolist()).lower()
        confidence = self.calculate_confidence(onnx_outputs)

        return transcription, confidence

    def calculate_confidence(self, onnx_outputs):
        """
        Calculate confidence from model outputs

        Args:
            onnx_outputs (np.ndarray): model outputs

        Returns:
            float: confidence score
        """
        softmax_outputs = torch.nn.functional.softmax(torch.tensor(onnx_outputs), dim=-1)
        confidence_scores = softmax_outputs.max(dim=-1)[0]
        confidence = confidence_scores.mean().item()

        return confidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wav2Vec2 ASR Model")
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--checkpoint_name", type=str, required=True)

    args = parser.parse_args()
    logger = get_logger()

    model = QuantizedWave2Vec2ForCTC(
        checkpoint_name=args.checkpoint_name,
        model_path=args.model_path
    )

    transcription, confidence = model.predict(args.audio_path)

    logger.info(f"Transcription: {transcription}, Confidence: {confidence}")
