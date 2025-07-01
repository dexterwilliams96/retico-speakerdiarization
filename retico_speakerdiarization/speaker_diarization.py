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

# Buffer for embeddings
# Every time a buffer is added check which cluster it belongs to
# If we have no clusters, assign a dummy one
# After computing 10 embeddings, compute optimal cluster number
# Compute centroids and reassign labels then commit
# Each new ten received, add labels based on existing centroids, compute new optimal number, compute new centroids
# When computing new centroids maintain previous labels
# Once we receive 100 embeddings, prune, delete embeddings but maintain relevant ones for each centroid, maintain at least 10 embeddings per centroid


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
        similiarity_threshold=0.8,
        upper_cluster_bound=5,
        max_k_means_iterations=10,
        max_buffer_size=100,
        min_buffer_size=10,
    ):
        # Initialize speaker embedding model
        self.device = device
        model_id = "pyannote/embedding"
        model = Model.from_pretrained(model_id)
        self.model = Inference(model, window="whole").to(
            torch.device(self.device))

        # Speaker embedding related
        self.embedding_buffer = []
        self.similiarity_threshold = similiarity_threshold
        self.use_k_means = False
        self.centroids = None
        self.upper_cluster_bound = upper_cluster_bound
        self.max_k_means_iterations = max_k_means_iterations
        self.max_buffer_size = max_buffer_size
        self.min_buffer_size = min_buffer_size

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

    def _compute_centroids(self, k):
        # K-means clustering over speaker embeddings
        embeddings = torch.stack(self.embedding_buffer).to(self.device)
        centroids = embeddings[torch.randperm(
            embeddings.size(0), device=self.device)[:k]]
        for _ in range(self.max_k_means_iterations):
            centroids_norm = normalize(centroids, dim=1)
            sim = torch.matmul(embeddings, centroids_norm.T)
            cluster_ids = sim.argmax(dim=1)
            for i in range(k):
                labels = embeddings[cluster_ids == i]
                if len(labels) > 0:
                    centroids[i] = labels.mean(dim=0)
        return centroids, cluster_ids

    def _map_cluster_ids(self, best_centroids, cluster_ids):
        all_scores = [[] for _ in range(len(self.centroids))]
        for old_id, old_centroid in enumerate(self.centroids):
            scores = [0 for _ in range(len(best_centroids))]
            for new_id, centroid in enumerate(best_centroids):
                sim = torch.dot(centroid, old_centroid)
                scores[new_id] = sim.item()
            all_scores[old_id] = scores
        # Get best score for each id
        used = set()
        cluster_mapping = [0 for _ in range(len(best_centroids))]
        for _ in range(len(self.centroids)):
            best_score = -float('inf')
            best_new_id = None
            best_old_id = None
            for old_id, scores in enumerate(all_scores):
                for new_id, score in enumerate(scores):
                    if new_id not in used and (best_new_id is None or score > best_score):
                        best_score = score
                        best_new_id = new_id
                        best_old_id = old_id
            if best_new_id is not None:
                used.add(best_new_id)
                cluster_mapping[best_new_id] = best_old_id
        # If any new centroids are not mapped, we should create new centroids
        diff = set((cluster_ids).tolist()) - used
        max_id = max(cluster_mapping)
        for new_id in diff:
            max_id += 1
            cluster_mapping[new_id] = max_id
        # Map cluster ids to new ids
        for new_id, old_id in enumerate(cluster_mapping):
            cluster_ids[cluster_ids == old_id] = cluster_mapping[new_id]
        return cluster_ids

    def get_best_centroids(self):
        coef = SilhouetteScore(device=self.device)
        best_centroids = None
        best_cluster_ids = None
        best_score = -1
        for k in range(2, self.upper_cluster_bound + 1):
            centroids, cluster_ids = self._compute_centroids(k)
            coef.reset()
            embeddings = torch.stack(self.embedding_buffer).to(self.device)
            coef.update((embeddings, cluster_ids))
            score = coef.compute()
            if best_centroids is None or score > best_score:
                best_score = score
                best_centroids = centroids
                best_cluster_ids = cluster_ids
        if self.centroids is not None:
            best_cluster_ids = self._map_cluster_ids(
                best_centroids, best_cluster_ids)
        self.centroids = best_centroids
        return best_cluster_ids

    def _prune_embeddings(self):
        # TODO cut down embeddings to size, maintain most important
        # Keep 90 most recent embeddings
        self.embedding_buffer = self.embedding_buffer[-90:]

    def _add_embedding(self, embedding):
        if len(self.embedding_buffer) >= self.max_buffer_size:
            self.prune_embeddings()
        self.embedding_buffer.append(embedding)

    def _get_speaker_id(self, embedding):
        if self.centroids is None:
            return 0
        sim = torch.matmul(self.centroids, embedding)
        return sim.argmax().item()

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
        transcription = None
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

        # Get embedding and normalize
        embedding = normalize(torch.from_numpy(self.model({"waveform": torch.from_numpy(
            npa).unsqueeze(0), "sample_rate": self.framerate})), dim=0).to(self.device)
        self._add_embedding(embedding)
        prediction = self._get_speaker_id(embedding)
        # Check if we need to use k-means clustering
        if not self.use_k_means and len(self.embedding_buffer) > 1:
            embeddings = torch.stack(self.embedding_buffer).to(self.device)
            sim = (embeddings.sum() - embeddings.diag().sum()) / \
                (len(embeddings)**2 - len(embeddings))
            if sim < self.similiarity_threshold:
                self.use_k_means = True

        if silence:
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

            output_iu = self.create_iu(self.latest_input_iu)
            output_iu.set_speaker(prediction)
            self.current_output.append(output_iu)
            um = retico_core.UpdateMessage()
            um.add_iu(output_iu, retico_core.UpdateType.ADD)

            if len(self.current_output) == self.sd.min_buffer_size:
                # If we are using k-means, update centroids
                if self.sd.use_k_means:
                    cluster_ids = self.sd.get_best_centroids()
                    for i, iu in enumerate(self.current_output):
                        speaker_id = cluster_ids[len(
                            self.sd.embedding_buffer) - (1 + i)]
                        if i > 0 and i < len(self.current_output) - 1:
                            # Lazy: assume if only a single chunk is different, it is the same speaker
                            if cluster_ids[i - 1] != speaker_id and cluster_ids[i + 1] != speaker_id:
                                speaker_id = cluster_ids[i - 1]
                        if speaker_id != iu.speaker:
                            new_iu = self.create_iu(iu)
                            self.revoke(iu)
                            um.add_iu(iu, retico_core.UpdateType.REVOKE)
                            new_iu.set_speaker(speaker_id)
                            self.commit(new_iu)
                            um.add_iu(new_iu, retico_core.UpdateType.COMMIT)
                        else:
                            self.commit(iu)
                            um.add_iu(iu, retico_core.UpdateType.COMMIT)
                for iu in self.current_output:
                    self.commit(iu)
                    um.add_iu(iu, retico_core.UpdateType.COMMIT)
                self.current_output = []

            self.latest_input_iu = None
            self.append(um)

    def prepare_run(self):
        self._sd_thread_active = True
        # Make thread daemon for cleaner shutdown
        threading.Thread(target=self._sd_thread, daemon=True).start()

    def shutdown(self):
        self._sd_thread_active = False
        self.sd.reset()
