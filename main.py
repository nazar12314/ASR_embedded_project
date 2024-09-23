import warnings
import argparse
import sounddevice as sd
import numpy as np
import Levenshtein
import textwrap
import concurrent.futures
import socket

from core.asr_model import QuantizedWave2Vec2ForCTC
from core.validator import VoiceValidator
from core.database import FaissDatabase, UserDatabase

from utils.config_utils import load_config
from utils.logger_utils import get_logger


logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")


class SecuritySystem:
    def __init__(self, config, remote=False):
        self.model = QuantizedWave2Vec2ForCTC(**config["model"])

        self.faiss_database = FaissDatabase(**config["faiss_db"])
        self.user_database = UserDatabase(**config["sqlite_db"])

        self.VoiceValidator = VoiceValidator(self.faiss_database)

        self.get_voice_input = self.get_voice_input_remote if remote else self.get_voice_input_local

        self.passphrase_threshold = 0.33

    def register_user(self):
        """
        Register a new user by recording their passphrase and storing their voice embedding.
        """
        logger.info("Registering user...")

        passphrase = self.get_voice_input("Please say your passphrase: ")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_embedding = executor.submit(self.VoiceValidator.embedd_audio, passphrase)
            future_transcription = executor.submit(self.model.predict, passphrase)

            voice_embedding = future_embedding.result()
            passphrase_transcription, _ = future_transcription.result()

        logger.info(f"Passphrase: {passphrase_transcription}")

        embedding_id = self.faiss_database.add_embedding(voice_embedding)

        user_id = self.user_database.insert_user(passphrase_transcription, embedding_id)
        logger.info(f"User registered. User ID: {user_id}")

        self.faiss_database.save_index()

    def validate_user(self):
        """
        Validate a user by recording their passphrase and comparing it with the stored voice embedding.

        Returns:
            bool: True if the user is validated, False otherwise
        """
        user_id = input("Please write your id: ")

        user = self.user_database.get_user_by_id(user_id)

        if user is None:
            logger.info("User not found.")

        passphrase = self.get_voice_input("Please say your passphrase: ")

        logger.info("Validating user...")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_embedding = executor.submit(self.VoiceValidator.embedd_audio, passphrase)
            future_transcription = executor.submit(self.model.predict, passphrase)

            voice_embedding = future_embedding.result()
            passphrase_transcription, _ = future_transcription.result()

        if not self.VoiceValidator.validate(voice_embedding, user.faiss_index):
            logger.info("User not recognized... Validation failed.")
            return

        if not self.validate_passphrase(passphrase_transcription, user.passphrase):
            logger.info("Passphrase not recognized... Validation failed.")
            return
        
        logger.info("User was successfully validated.")

    def launch(self):
        """
        Launch the security system and provide the user with options to register or validate a user.
        """
        options = {
            "1": self.register_user,
            "2": self.validate_user,
            "0": lambda: None
        }

        while True:
            option = input(
                textwrap.dedent("""\n\
                    Voice Security System 
                    1. Register user 
                    2. Validate user 
                    0. Exit 
                    Please select an option: """)
            )

            if option not in options:
                logger.info("Invalid option. Please try again.")
                continue

            if option == "0":
                logger.info("Exiting...")
                break

            options[option]()

        self.user_database.close()

    def get_voice_input_local(self, prompt: str, duration: float = 5.0, sample_rate: int = 16000):
        """
        Record audio input from the user.

        Args:
            prompt (str): The prompt message to display to the user.
            duration (float): The duration in seconds to record audio.
            sample_rate (int): The sample rate to record audio.

        Returns:
            np.ndarray: The recorded audio data.
        """
        logger.info(prompt)

        logger.info(f"Recording for {duration} seconds...")
        
        try:
            audio_data = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype='float32'
            )

            sd.wait()
            logger.info("Recording finished.")

            audio_data_int16 = np.int16(audio_data * 32767)

            return audio_data_int16.squeeze()
        except Exception as e:
            logger.error(f"Error while recording audio: {e}")
            return None
        
    def get_voice_input_remote(self, prompt: str, duration: float = 5.0, sample_rate: int = 16000):
        logger.info(prompt)

        mac_ip = "10.10.229.27"
        mac_port = 12345

        logger.info(f"Connecting to Mac microphone at {mac_ip}:{mac_port}")

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((mac_ip, mac_port))

            audio_size = int(duration * sample_rate)
            audio_data = sock.recv(audio_size * 2)
            sock.close()

            audio_data = np.frombuffer(audio_data, dtype=np.int16)
            logger.info("Audio received.")
            return audio_data
        except Exception as e:
            logger.error(f"Error while receiving audio from Mac: {e}")
            return None

    def validate_passphrase(self, input_passphrase, stored_passphrase):
        distance = Levenshtein.distance(input_passphrase, stored_passphrase)

        max_length = max(len(input_passphrase), len(stored_passphrase))
    
        if max_length == 0:
            normalized_distance = 0
        else:
            normalized_distance = distance / max_length

        logger.info(f"Normalized Levenshtein distance: {normalized_distance}")

        return normalized_distance <= self.passphrase_threshold


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Security System")
    parser.add_argument("--config", type=str, required=True)

    args = parser.parse_args()

    config = load_config(args.config, logger=logger)

    security_system = SecuritySystem(config)

    security_system.launch()
