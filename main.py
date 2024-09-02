from asr import QuantizedWave2Vec2ForCTC


def main():
    model = QuantizedWave2Vec2ForCTC(
        "facebook/wav2vec2-base-960h",
        "output/model.quant.onnx"
    )

    transcription, confidence = model.predict("audios/rec5.wav")

    print(f"Transcription: {transcription}")
    print(f"Confidence: {confidence}")


if __name__ == "__main__":
    main()
