import difflib
import warnings

from asr import QuantizedWave2Vec2ForCTC
from validate_voice import VoiceValidator
from databases.faiss_database import FaissDatabase
from databases.sqlite3_database import SQLite3Database


warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")


class SecuritySystem:
    def __init__(self):
        self.model = QuantizedWave2Vec2ForCTC(
            "facebook/wav2vec2-base-960h",
            "output/model.quant.onnx"
        )

        self.VoiceValidator = VoiceValidator("output/embeddings.npy")
        self.faiss_database = FaissDatabase()
        self.user_database = SQLite3Database("data/users.db")

        self.confidence_threshold = 0.85
        self.passphrase_threshold = 0.7

    def create_user(self):
        name = input("\nPlease say your name: ")
        surname = input("Please say your surname: ")
        passphrase = self.voice_to_text("Please say your passphrase: ")

        voice_embedding = self.VoiceValidator.embedd_audio(passphrase)
        passphrase_transcription, confidence = self.model.predict(passphrase)

        embedding_id = self.faiss_database.add_embedding(voice_embedding)

        self.user_database.execute(
            f"INSERT INTO users (name, surname, passphrase, faiss_index_id)"
            f" VALUES ('{name}', '{surname}', '{passphrase_transcription}', {embedding_id})"
        )

        print(f"\nUser {name} {surname} created with index {embedding_id}.\n")

        self.faiss_database.save_index()

    def validate_user(self):
        user_id = input("Please say your id: ")
        passphrase = self.voice_to_text("Please say your passphrase: ")

        user = self.user_database.execute(
            f"SELECT * FROM users WHERE id = {user_id}", fetch=True
        )

        if user is None or len(user) == 0:
            return False

        print(user)

        voice_embedding = self.VoiceValidator.embedd_audio(passphrase)
        passphrase_transcription, confidence = self.model.predict(passphrase)

        if confidence < self.confidence_threshold:
            print(f"\nConfidence too low. Confidence: {confidence}")
            return False

        distances, indices = self.faiss_database.search_embedding(voice_embedding)

        if indices[0][0] != user[0][4]:
            print("\nVoice not recognized.")
            return False

        if not self.validate_passphrase(passphrase_transcription, user[0][3]):
            print("\nPassphrase not recognized.")
            return False

        return True

    def run(self):
        while True:
            option = input("Security System \n1. Create user\n2. Validate user\nPlease select an option: ")

            if option == "1":
                self.create_user()
            elif option == "2":
                validated = self.validate_user()

                if validated:
                    print("User validated.\n")
                else:
                    print("User not validated.\n")
            else:
                print("Invalid option. Exiting.")
                break

        self.user_database.close()

    def voice_to_text(self, prompt):
        print(prompt)
        return input("Path to the audio file: ")

    def validate_passphrase(self, input_passphrase, stored_passphrase):
        similarity = difflib.SequenceMatcher(None, input_passphrase, stored_passphrase).ratio()
        print(f"Passphrase similarity: {similarity}")
        return similarity >= self.passphrase_threshold


def main():
    security_system = SecuritySystem()
    security_system.run()


if __name__ == "__main__":
    main()
