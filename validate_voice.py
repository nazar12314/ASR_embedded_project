import numpy as np
import torch

from resemblyzer import preprocess_wav, VoiceEncoder


class VoiceValidator:
    def __init__(self, embedding_db):
        self.encoder = VoiceEncoder(verbose=False)
        self.embedding_db = embedding_db

    def embedd_audio(self, file_path: str) -> np.ndarray:
        """Generating embedding for audio sample

        Args:
            file_path (str): path to audio file

        Returns:
            np.ndarray: generated embedding
        """
        wav = preprocess_wav(file_path)

        embedding = self.encoder.embed_utterance(wav)

        return embedding

    def validate(self, voice_audio):
        pass


def embedd_audio(file_path: str) -> np.ndarray:
    """Generating embedding for audio sample

    Args:
        file_path (str): path to audio file

    Returns:
        np.ndarray: generated embedding
    """
    encoder = VoiceEncoder(verbose=False)
    wav = preprocess_wav(file_path)

    embedding = encoder.embed_utterance(wav)

    return embedding


if __name__ == "__main__":
    first_file = "audios/rec1.wav"
    second_file = "audios/harvard.wav"

    embedding1 = embedd_audio(first_file)
    embedding2 = embedd_audio(second_file)

    tensor1 = torch.tensor(embedding1).unsqueeze(0)
    tensor2 = torch.tensor(embedding2).unsqueeze(0)

    # Calculate cosine similarity
    similarity = torch.nn.functional.cosine_similarity(tensor1, tensor2).item()

    print(similarity)

