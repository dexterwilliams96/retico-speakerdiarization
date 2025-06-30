import numpy as np
import pydub
import retico_core
import threading
import time
import torch
import webrtcvad

from pyannote.audio import Inference, Model
from retico_core.audio import AudioIU
from retico_core.text import SpeechRecognitionIU
from scipy.spatial.distance import cosine
from torch.nn.functional import normalize


class SpeakerDiarization:
    def __init__(
        self,
        framerate=16_000,
        silence_dur=1,
        vad_agressiveness=3,
        silence_threshold=0.75,
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_new_tokens=256,
        use_cache=True,
        similiarity_threshold=0.5,
    ):
        self.device = device
        model_id = "pyannote/embedding"
        model = Model.from_pretrained(model_id)
        self.model = Inference(model, window="whole").to(
            torch.device(self.device))
        self.speakers = {}
        self.similiarity_threshold = similiarity_threshold
        self.audio_buffer = []
        self.framerate = framerate
        self.vad = webrtcvad.Vad(vad_agressiveness)
        self.silence_dur = silence_dur
        self.vad_state = False
        self._n_sil_frames = None
        self.silence_threshold = silence_threshold
        self.max_new_tokens = max_new_tokens
        self._temp_audio_array = None

    def _resample_audio(self, audio):
        if self.framerate != 16_000:
            # resample if framerate is not 16 kHz
            s = pydub.AudioSegment(
                audio, sample_width=2, channels=1, frame_rate=self.framerate
            )
            s = s.set_frame_rate(16_000)
            return s._data
        return audio

    def get_n_sil_frames(self):
        if not self._n_sil_frames:
            if len(self.audio_buffer) == 0:
                return None
            frame_length = len(self.audio_buffer[0]) / 2
            self._n_sil_frames = int(
                self.silence_dur / (frame_length / 16_000))
        return self._n_sil_frames

    def recognize_silence(self):
        n_sil_frames = self.get_n_sil_frames()
        if not n_sil_frames or len(self.audio_buffer) < n_sil_frames:
            return True
        silence_counter = 0
        for a in self.audio_buffer[-n_sil_frames:]:
            if not self.vad.is_speech(a, 16_000):
                silence_counter += 1
        if silence_counter >= int(self.silence_threshold * n_sil_frames):
            return True
        return False

    def add_audio(self, audio):
        audio = self._resample_audio(audio)
        self.audio_buffer.append(audio)

    def recognize(self):
        silence = self.recognize_silence()
        transcription = None
        if not self.vad_state and not silence:
            self.vad_state = True
            self.audio_buffer = self.audio_buffer[-self.get_n_sil_frames():]

        if not self.vad_state:
            return None, False

        if len(self.audio_buffer) == 0:
            return None, False

        total_length = sum(len(a) for a in self.audio_buffer)
        if total_length < 10:
            return None, False

        audio_arrays = [np.frombuffer(a, dtype=np.int16)
                        for a in self.audio_buffer]
        full_audio_np = np.concatenate(audio_arrays)
        npa = full_audio_np.astype(np.float32) / 32768.0
        # Get embedding and normalize
        embedding = normalize(torch.from_numpy(self.model({"waveform": torch.from_numpy(
            npa).unsqueeze(0), "sample_rate": self.framerate})), dim=0)
        # If there are no speakers, add the first one
        if len(self.speakers) == 0:
            self.speakers[f'SPEAKER_{len(self.speakers)}'] = {
                "embedding": embedding, "count": 1}
        else:
            best_match = None
            min_distance = float('inf')
            for speaker, data in self.speakers.items():
                dist = cosine(embedding, data["embedding"])
                if dist < min_distance:
                    min_distance = dist
                    best_match = speaker
            if min_distance < self.similiarity_threshold:
                # Compute running average
                self.speakers[best_match]["count"] += 1
                self.speakers[best_match]["embedding"] = (
                    (self.speakers[best_match]["embedding"] *
                     self.speakers[best_match]["count"]) + embedding
                ) / (self.speakers[best_match]["count"] + 1)
                # Normalize the embedding
                self.speakers[best_match]["embedding"] = normalize(
                    self.speakers[best_match]["embedding"], dim=0)
            else:
                self.speakers[f'SPEAKER_{len(self.speakers)}'] = {
                    "embedding": embedding,
                    "count": 1,
                }
        print(self.speakers.keys())
        if silence:
            self.vad_state = False
            self.audio_buffer = []
        return transcription, self.vad_state

    def reset(self):
        self.vad_state = True
        self.audio_buffer = []


class SpeakerDiarizationModule(retico_core.AbstractModule):
    @staticmethod
    def name():
        return "Speaker Diarization Module"

    @staticmethod
    def description():
        return "A module that recognizes speakers in audio input."

    @staticmethod
    def input_ius():
        return [AudioIU]

    @staticmethod
    def output_iu():
        return SpeechRecognitionIU

    def __init__(self, framerate=None, silence_dur=1, **kwargs):
        super().__init__(**kwargs)

        self.asr = SpeakerDiarization(
            silence_dur=silence_dur,
        )
        self.framerate = framerate
        self.silence_dur = silence_dur
        self._asr_thread_active = False
        self.latest_input_iu = None

    def process_update(self, update_message):
        for iu, ut in update_message:
            # Audio IUs are only added and never updated.
            if ut != retico_core.UpdateType.ADD:
                continue
            if self.framerate is None:
                self.framerate = iu.rate
                self.asr.framerate = self.framerate
            self.asr.add_audio(iu.raw_audio)
            if not self.latest_input_iu:
                self.latest_input_iu = iu

    def _asr_thread(self):
        while self._asr_thread_active:
            time.sleep(0.5)
            if not self.framerate:
                continue
            prediction, vad = self.asr.recognize()
            if prediction is None:
                continue
            end_of_utterance = not vad
            um, new_tokens = retico_core.text.get_text_increment(
                self, prediction)

            if len(new_tokens) == 0 and vad:
                continue

            for i, token in enumerate(new_tokens):
                output_iu = self.create_iu(self.latest_input_iu)
                eou = i == len(new_tokens) - 1 and end_of_utterance
                output_iu.set_asr_results([prediction], token, 0.0, 0.99, eou)
                self.current_output.append(output_iu)
                um.add_iu(output_iu, retico_core.UpdateType.ADD)

            if end_of_utterance:
                for iu in self.current_output:
                    self.commit(iu)
                    um.add_iu(iu, retico_core.UpdateType.COMMIT)
                self.current_output = []

            self.latest_input_iu = None
            self.append(um)

    def prepare_run(self):
        self._asr_thread_active = True
        # Make thread daemon for cleaner shutdown
        threading.Thread(target=self._asr_thread, daemon=True).start()

    def shutdown(self):
        self._asr_thread_active = False
        self.asr.reset()
