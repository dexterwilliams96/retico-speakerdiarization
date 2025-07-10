import glob
import numpy as np
import os
import pydub
import random
import retico_core
import threading
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import webrtcvad

from retico_core import abstract
from retico_core.audio import AudioIU
from retico_core.text import SpeechRecognitionIU
from speechbrain.inference.speaker import SpeakerRecognition


class SpeakerIU(abstract.IncrementalUnit):
    """An Incremental Unit representing a speaker in the audio stream."""

    @staticmethod
    def type():
        return "Speaker Incremental Unit"

    def __init__(self, creator=None, iuid=0, previous_iu=None, grounded_in=None,
                 payload=None, decision=None, speaker=None, embedding=None, **kwargs):
        super().__init__(creator=creator, iuid=iuid, previous_iu=previous_iu,
                         grounded_in=grounded_in, payload=payload, decision=decision, **kwargs)
        self.speaker = speaker
        self.embedding = embedding

    def get_speaker(self):
        return self.speaker

    def set_speaker(self, speaker):
        """Set the speaker of the IU.

        Args:
            speaker: Tuple, with first index as speaker name (or None if can't be matched)
                     and second index as boolean representing confirmation/commit.
        """
        self.speaker = speaker
        self.payload = {'speaker': speaker}

    def get_embedding(self):
        """Get the embedding of the IU.

        Returns:
            torch.Tensor: The embedding of the speaker.
        """
        return self.embedding

    def set_embedding(self, embedding):
        """Set the embedding of the IU.

        Args:
            embedding (torch.Tensor): The embedding of the speaker.
        """
        self.embedding = embedding

    def __repr__(self):
        return f"{self.type()} - {self.creator.name()}: {self.payload['speaker']}"


class CosineSimClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


class SpeakerDiarization:
    def __init__(
        self,
        framerate=16_000,
        silence_dur=1,
        vad_agressiveness=3,
        silence_threshold=0.75,
        device="cpu",
        sceptical_threshold=0.4,
        train_buffer_size=1,
        audio_path='audio',
        compile_model=True
    ):
        # Initialize speaker embedding model
        self.device = torch.device(device)
        self.model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa")
        self.model.device = device
        for module in self.model.modules():
            module.to(device)
        if compile_model:
            self.model = torch.compile(self.model)
        self.classifier = CosineSimClassifier().to(device)
        self.optimizer = torch.optim.SGD(self.classifier.parameters(), lr=0.1, weight_decay=1e-4)
        self.loss_fn = nn.BCELoss()

        # Speaker embedding related
        self.speaker_1_embedding = None
        self.speaker_2_embedding = None
        self.speaker_1_name = None
        self.speaker_2_name = None
        self.speaker_map = dict()
        self.sceptical_threshold = sceptical_threshold
        self.audio_path = audio_path

        # Audio related
        self.audio_buffer = []
        self.framerate = framerate
        self.vad = webrtcvad.Vad(vad_agressiveness)
        self.silence_dur = silence_dur
        self.vad_state = False
        self._n_sil_frames = None
        self.silence_threshold = silence_threshold

        # Setup
        self.train_buffer_size = train_buffer_size
        self._get_initial_centroids()
        self.data_s1 = self._generate_synthetic_data(
            self.speaker_1_embedding, self.speaker_1_name)
        self.data_s2 = self._generate_synthetic_data(
            self.speaker_2_embedding, self.speaker_2_name)
        self._train_classifier()

    def _get_initial_centroids(self):
        # Should be two of these
        audio_files = glob.glob(f"{self.audio_path}/*")
        with torch.no_grad():
            self.speaker_1_name = os.path.splitext(
                os.path.basename(audio_files[0]))[0]
            wav, fs = torchaudio.load(audio_files[0])
            self.speaker_1_embedding = F.normalize(self.model.encode_batch(
                wav).squeeze(0).squeeze(0), dim=0).to(self.device)
            self.speaker_2_name = os.path.splitext(
                os.path.basename(audio_files[1]))[0]
            wav, fs = torchaudio.load(audio_files[1])
            self.speaker_2_embedding = F.normalize(self.model.encode_batch(
                wav).squeeze(0).squeeze(0), dim=0).to(self.device)
        self.speaker_map[self.speaker_1_name] = 0.9
        self.speaker_map[self.speaker_2_name] = 0.1

    def _generate_synthetic_data(self, embedding, label, num_samples=20, noise_std=0.01):
        data = []
        emb = embedding.unsqueeze(0)
        for _ in range(num_samples):
            noisy = F.normalize(emb + torch.randn_like(emb)
                                * noise_std, dim=1).to(self.device)
            data.append((noisy, torch.tensor(
                [self.speaker_map[label]], dtype=torch.float32).to(self.device)))
        return data

    def _train_classifier(self, epochs=50, sample_size=10):
        for _ in range(epochs):
            batch = random.sample(self.data_s1, sample_size) + \
                random.sample(self.data_s2, sample_size)
            random.shuffle(batch)
            for emb, label in batch:
                sim_1 = F.cosine_similarity(emb, self.speaker_1_embedding)
                sim_2 = F.cosine_similarity(emb, self.speaker_2_embedding)
                input_vec = torch.tensor(
                    [[sim_1.item(), sim_2.item()]]).to(self.device)
                output = self.classifier(input_vec)
                target = torch.tensor([[label]], dtype=torch.float32).to(self.device)
                loss = self.loss_fn(output, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        self.data_s1 = []
        self.data_s2 = []

    def _classify_embedding(self, embedding):
        sim_1 = F.cosine_similarity(embedding, self.speaker_1_embedding, dim=0)
        sim_2 = F.cosine_similarity(embedding, self.speaker_2_embedding, dim=0)
        input_vec = torch.tensor([[sim_1.item(), sim_2.item()]]).to(self.device)
        prob_1 = self.classifier(input_vec).item()
        return self.speaker_1_name if prob_1 >= 0.5 else self.speaker_2_name, 1.0 - prob_1

    def add_embedding(self, embedding):
        label, confidence = self._classify_embedding(embedding)
        # If the speaker can be strongly confirmed
        if confidence >= self.sceptical_threshold:
            if label == self.speaker_1_name:
                self.data_s1.append(
                    (embedding, torch.tensor([self.speaker_map[label]], dtype=torch.float32)))
            else:
                self.data_s2.append(
                    (embedding, torch.tensor([self.speaker_map[label]], dtype=torch.float32)))
            if len(self.data_s1) == self.train_buffer_size and len(self.data_s2) == self.train_buffer_size:
                self._train_classifier(epochs=1, sample_size=self.train_buffer_size)
            return label, True
        # Weak confirmation
        return label, False

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
        embedding = None
        if not self.vad_state and not silence:
            self.vad_state = True
            self.audio_buffer = self.audio_buffer[-self._get_n_sil_frames():]

        if not self.vad_state:
            return None, False, None

        if len(self.audio_buffer) == 0:
            return None, False, None

        total_length = sum(len(a) for a in self.audio_buffer)
        if total_length < 10:
            return None, False, None

        audio_arrays = [np.frombuffer(a, dtype=np.int16)
                        for a in self.audio_buffer]
        full_audio_np = np.concatenate(audio_arrays)
        npa = full_audio_np.astype(np.float32) / 32768.0
        npa = torch.from_numpy(
            np.clip(npa, -1, 1).astype(np.float32)).to(self.device)
        if silence:
            # Get embedding and F.normalize
            with torch.no_grad():
                embedding = F.normalize(self.model.encode_batch(
                    npa.unsqueeze(0)).squeeze(0).squeeze(0), dim=0).to(self.device)
                prediction = self.add_embedding(embedding)
                self.vad_state = False
                self.audio_buffer = []

        return prediction, self.vad_state, embedding

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

    def __init__(self, framerate=None, silence_dur=1,
                 sceptical_threshold=0.4,
                 # Specify a path to speaker recordings, if you want initial centroids
                 audio_path='audio',
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 **kwargs):
        super().__init__(**kwargs)

        self.sd = SpeakerDiarization(
            silence_dur=silence_dur,
            sceptical_threshold=sceptical_threshold,
            audio_path=audio_path,
            device=device
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
            prediction, vad, embedding = self.sd.recognize()
            if prediction is None:
                continue
            speaker, confirmed = prediction
            end_of_utterance = not vad

            output_iu = self.create_iu(self.latest_input_iu)
            output_iu.set_speaker(prediction)
            um = retico_core.UpdateMessage()
            # If the speaker is confirmed commit the IU, and check all unconfirmed IUs
            remove_set = []
            if confirmed:
                um.add_iu(output_iu, retico_core.UpdateType.COMMIT)
                remove_set.append(output_iu)
                for iu in self.current_output:
                    if not iu.committed:
                        prediction = self.sd.add_embedding(
                            iu.get_embedding())
                        speaker, confirmed = prediction
                        new_iu = self.create_iu(
                            iu.grounded_in) if speaker != iu.speaker[0] else iu
                        if confirmed:
                            um.add_iu(new_iu, retico_core.UpdateType.COMMIT)
                            remove_set.append(new_iu)
                        elif speaker != iu.speaker[0]:
                            new_iu.set_speaker(prediction)
                            um.add_iu(iu, retico_core.UpdateType.REVOKE)
                            remove_set.append(iu)
                            new_iu.set_embedding(iu.get_embedding())
                            um.add_iu(new_iu, retico_core.UpdateType.ADD)
                            self.current_output.append(new_iu)
            # If the speaker is not confirmed, add the IU to the current output
            else:
                output_iu.set_embedding(embedding)
                self.current_output.append(output_iu)
                um.add_iu(output_iu, retico_core.UpdateType.ADD)
            self.current_output = [
                iu for iu in self.current_output if iu not in remove_set]
            self.latest_input_iu = None
            self.append(um)

    def prepare_run(self):
        self._sd_thread_active = True
        # Make thread daemon for cleaner shutdown
        threading.Thread(target=self._sd_thread, daemon=True).start()

    def shutdown(self):
        self._sd_thread_active = False
        self.sd.reset()
