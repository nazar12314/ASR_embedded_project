from transformers import Wav2Vec2Processor
import torch
import librosa
import onnxruntime as rt
import numpy as np


class QuantizedWave2Vec2ForCTC:
    def __init__(self, checkpoint_name, model_path):
        options = rt.SessionOptions()
        options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.model = rt.InferenceSession(model_path, options)
        self.processor = Wav2Vec2Processor.from_pretrained(checkpoint_name)

    def predict(self, audio_path):
        audio, rate = librosa.load(audio_path, sr=16000)
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
        softmax_outputs = torch.nn.functional.softmax(torch.tensor(onnx_outputs), dim=-1)
        confidence_scores = softmax_outputs.max(dim=-1)[0]
        confidence = confidence_scores.mean().item()

        return confidence
