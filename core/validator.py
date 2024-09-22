import numpy as np

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

    def validate(self, audio_embedding: np.ndarray, index_id: int) -> bool:
        """
        Validate if the audio embedding is the same as the one in the database

        Args:
            audio_embedding (np.ndarray): audio embedding
            index_id (int): index id of the audio embedding in the database

        Returns:
            bool: True if the audio embedding is the same as the one in the database
        """
        _, indices = self.embedding_db.search_embedding(audio_embedding)

        closest_index = indices[0][0]

        return closest_index == index_id
