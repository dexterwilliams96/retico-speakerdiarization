import pyaudio
import wave

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16_000
RECORD_SECONDS = 10
WAVE_OUTPUT_FILENAME = "output.wav"

def record_audio(speaker_name):
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK)
    print("Recording...")
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)
    print("Recording finished.")
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf = wave.open(f"{speaker_name}.wav", 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    print(f"Audio saved to {speaker_name}.wav")

num_speakers = int(input("Enter number of speakers"))
for i in range(num_speakers):
    speaker_name = input(f"Enter name of speaker {i + 1}:")
    record_audio(speaker_name)
