import numpy as np
import pydub
import retico_core
import threading
import time
import torch
import webrtcvad

from ignite.metrics.clustering import SilhouetteScore
from pyannote.audio import Inference, Model
from retico_core import abstract
from retico_core.audio import AudioIU
from retico_core.text import SpeechRecognitionIU
from torch.nn.functional import normalize

class SpeakerIU(abstract.IncrementalUnit):
    """An Incremental Unit representing a speaker in the audio stream."""

    @staticmethod
    def type():
        return "Speaker Incremental Unit"

    def __init__(self, creator=None, iuid=0, previous_iu=None, grounded_in=None,
                 payload=None, decision=None, speaker=None, **kwargs):
        super().__init__(creator=creator, iuid=iuid, previous_iu=previous_iu,
                         grounded_in=grounded_in, payload=payload, decision=decision, **kwargs)
        self.speaker = speaker

    def set_speaker(self, speaker):
        """Set the speaker of the IU.

        Args:
            speaker (int): The ID of the speaker.
        """
        self.speaker = speaker
        self.payload = {'speaker': speaker}

    def __repr__(self):
        return f"{self.type()} - {self.creator.name()}: {self.payload['speaker']}"


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
        num_speakers=2,
        similiarity_threshold=0.5,
        max_buffer_size=1000,
    ):
        # Initialize speaker embedding model
        self.device = device
        model_id = "pyannote/embedding"
        model = Model.from_pretrained(model_id)
        self.model = Inference(model, window="whole").to(
            torch.device(self.device))

        # Speaker embedding related
        self.centroids = dict()
        self.similiarity_threshold = similiarity_threshold
        self.num_speakers = num_speakers
        self.max_buffer_size = max_buffer_size

        # Audio related
        self.audio_buffer = []
        self.framerate = framerate
        self.vad = webrtcvad.Vad(vad_agressiveness)
        self.silence_dur = silence_dur
        self.vad_state = False
        self._n_sil_frames = None
        self.silence_threshold = silence_threshold
        self.max_new_tokens = max_new_tokens
        self._temp_audio_array = None

    def _prune_embeddings(self):
        # TODO cut down embeddings to size, maintain most important
        pass

    def _add_embedding(self, embedding):
        if len(self.centroids) == 0:
            self.centroids["SPEAKER 0"] = [embedding, 1]
            return "SPEAKER 0"
        else:
            best_sim = -1
            best_speaker = None
            for speaker, (centroid, count) in self.centroids.items():
                sim = torch.cosine_similarity(embedding, centroid, dim=0)
                if sim > best_sim:
                    best_sim = sim
                    best_speaker = speaker
            best_centroid = self.centroids[best_speaker][0]
            best_count = self.centroids[best_speaker][1]
            if best_sim >= self.similiarity_threshold or len(self.centroids) == self.num_speakers:
                self.centroids[best_speaker][1] = best_count + 1
                self.centroids[best_speaker][0] = normalize((best_count * best_centroid + embedding) / (best_count + 1), dim=0)
                return speaker
            # If no similar speaker found, create a new one
            new_speaker_id = f"SPEAKER {len(self.centroids)}"
            self.centroids[new_speaker_id] = [embedding, 1]
            return new_speaker_id

    def _resample_audio(self, audio):
        if self.framerate != 16_000:
            # resample if framerate is not 16 kHz
            s = pydub.AudioSegment(
                audio, sample_width=2, channels=1, frame_rate=self.framerate
            )
            s = s.set_frame_rate(16_000)
            return s._data
        return audio

    def _get_n_sil_frames(self):
        if not self._n_sil_frames:
            if len(self.audio_buffer) == 0:
                return None
            frame_length = len(self.audio_buffer[0]) / 2
            self._n_sil_frames = int(
                self.silence_dur / (frame_length / 16_000))
        return self._n_sil_frames

    def _recognize_silence(self):
        n_sil_frames = self._get_n_sil_frames()
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
        silence = self._recognize_silence()
        prediction = None
        if not self.vad_state and not silence:
            self.vad_state = True
            self.audio_buffer = self.audio_buffer[-self._get_n_sil_frames():]

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


        if silence:
            # Get embedding and normalize
            embedding = normalize(torch.from_numpy(self.model({"waveform": torch.from_numpy(
                npa).unsqueeze(0), "sample_rate": self.framerate})), dim=0).to(self.device)
            prediction = self._add_embedding(embedding)
            self.vad_state = False
            self.audio_buffer = []

        return prediction, self.vad_state

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
        return SpeakerIU

    def __init__(self, framerate=None, silence_dur=1, **kwargs):
        super().__init__(**kwargs)

        self.sd = SpeakerDiarization(
            silence_dur=silence_dur,
        )
        self.framerate = framerate
        self.silence_dur = silence_dur
        self._sd_thread_active = False
        self.latest_input_iu = None

    def process_update(self, update_message):
        for iu, ut in update_message:
            # Audio IUs are only added and never updated.
            if ut != retico_core.UpdateType.ADD:
                continue
            if self.framerate is None:
                self.framerate = iu.rate
                self.sd.framerate = self.framerate
            self.sd.add_audio(iu.raw_audio)
            if not self.latest_input_iu:
                self.latest_input_iu = iu

    def _sd_thread(self):
        while self._sd_thread_active:
            time.sleep(0.5)
            if not self.framerate:
                continue
            prediction, vad = self.sd.recognize()
            if prediction is None:
                continue
            end_of_utterance = not vad

            output_iu = self.create_iu(self.latest_input_iu)
            output_iu.set_speaker(prediction)
            self.current_output.append(output_iu)
            um = retico_core.UpdateMessage()
            um.add_iu(output_iu, retico_core.UpdateType.COMMIT)
            self.latest_input_iu = None
            self.append(um)

    def prepare_run(self):
        self._sd_thread_active = True
        # Make thread daemon for cleaner shutdown
        threading.Thread(target=self._sd_thread, daemon=True).start()

    def shutdown(self):
        self._sd_thread_active = False
        self.sd.reset()
