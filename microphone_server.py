import socket
import sounddevice as sd
import numpy as np

from utils.logger_utils import get_logger


logger = get_logger(__name__)


def serve_audio(port=12345, duration=5.0, sample_rate=16000):
    """
    Serve audio data from the microphone to a remote client.

    Args:
        port (int): The port to listen on.
        duration (float): The duration in seconds to record audio.
        sample_rate (int): The sample rate to record audio
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('0.0.0.0', port))
    sock.listen(1)

    logger.info(f"Listening on port {port}...")

    while True:
        conn, addr = sock.accept()
        logger.info(f"Connection from {addr}")

        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()
        audio_data_int16 = np.int16(audio_data * 32767)

        conn.sendall(audio_data_int16.tobytes())
        conn.close()


if __name__ == "__main__":
    serve_audio()
