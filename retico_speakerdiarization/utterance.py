import retico_core

from retico_core.text import SpeechRecognitionIU
from retico_speakerdiarization import SpeakerIU
from sortedcontainers import SortedDict


class UtteranceIU(retico_core.IncrementalUnit):
    """An Incremental Unit representing an utterance from a speaker."""

    @staticmethod
    def type():
        return "Utterance Incremental Unit"

    def __init__(self, creator=None, iuid=0, previous_iu=None, grounded_in=None,
                 payload=None, decision=None, speaker=None, text=None, **kwargs):
        super().__init__(creator=creator, iuid=iuid, previous_iu=previous_iu,
                         grounded_in=grounded_in, payload=payload, decision=decision, **kwargs)
        self.speaker = speaker
        self.text = text
        self.payload = payload if payload is not None else {
            "speaker": self.speaker, "text": self.text}

    def get_speaker(self):
        return self.speaker

    def set_speaker(self, speaker):
        self.speaker = speaker
        self.payload["speaker"] = self.speaker

    def get_text(self):
        return self.text

    def set_text(self, text):
        self.text = text
        self.payload["text"] = self.text

    def __repr__(self):
        return f"Speaker {self.speaker}, Text: {self.text}"


class UtteranceModule(retico_core.AbstractModule):
    @staticmethod
    def name():
        return "Utterance Module"

    @staticmethod
    def description():
        return "A module that connects text to it's speaker."

    @staticmethod
    def input_ius():
        return [SpeechRecognitionIU, SpeakerIU]

    @staticmethod
    def output_iu():
        return UtteranceIU

    def __init__(self, merge_utterances=False, **kwargs):
        super().__init__(**kwargs)
        # Store a time-sorted list of utterances
        self.utterances = SortedDict()
        # Store a time-sorted list of speaker identificaitons
        self.speaker_timeline = SortedDict()
        # Maintain timestamp of last text
        self.last_text = None
        # Whether consective utterances from the same speaker should be merged
        self.merge_utterances = merge_utterances

    def _add_new_utterance(self, um, speaker, text, add_set=None):
        if add_set == None:
            add_set = self.current_output
        speaker_id, confirmed = speaker.get_speaker()
        if speaker_id is not None:
            output_iu = self.create_iu(speaker)
            output_iu.set_speaker(speaker_id)
            output_iu.set_text(text)
            if confirmed:
                um.add_iu(
                    output_iu, retico_core.UpdateType.COMMIT)
            else:
                um.add_iu(
                    output_iu, retico_core.UpdateType.ADD)
                add_set.append(output_iu)

    def _closer_to(self, utter_time, time1, time2):
        return abs(utter_time - time1) > abs(utter_time - time2)

    def _update_existing_ius(self, um, speakers_updated, old_last_speaker):
        new_last_speaker = self.speaker_timeline.keys()[
            len(self.speaker_timeline) - 1]
        # Update existing IUs
        remove_set = []
        add_set = []
        for iu in self.current_output:
            # Check the original audio in timestamp for the speaker id
            origin = iu.grounded_in.grounded_in.created_at
            # If this was an unconfirmed last speaker and the last speaker changed, recheck
            if self.last_text is not None and origin == old_last_speaker and new_last_speaker != old_last_speaker:
                if self._closer_to(self.last_text, old_last_speaker, new_last_speaker):
                    self.last_text = None
                    um.add_iu(iu, retico_core.UpdateType.REVOKE)
                    remove_set.append(iu)
                    speaker = self.speaker_timeline[new_last_speaker]
                    self._add_new_utterance(um, speaker, iu.get_text(), add_set)
                    # Will be revoked, don't do additional checks
                    continue
            # Otherwise if speaker changed
            if origin in speakers_updated:
                new_speaker = self.speaker_timeline[origin].get_speaker()
                new_iu = self.create_iu(
                    iu.grounded_in) if iu.get_speaker() != new_speaker[0] else iu
                # If the speaker id changed then revoke the IU
                if new_speaker[0] != iu.get_speaker():
                    um.add_iu(iu, retico_core.UpdateType.REVOKE)
                    remove_set.append(iu)
                # If speaker confirmed and not unknown commit the IU
                if new_speaker[0] is not None and new_speaker[1]:
                    um.add_iu(new_iu, retico_core.UpdateType.COMMIT)
                    remove_set.append(new_iu)
                # If the speaker id is uncomfirmed but known then add the IU
                elif new_speaker[0] is not None and not new_speaker[1]:
                    um.add_iu(new_iu, retico_core.UpdateType.ADD)
                    add_set.append(new_iu)
        self.current_output = [
            iu for iu in self.current_output if iu not in remove_set]
        self.current_output = self.current_output + add_set

    def _create_new_utterances(self, um):
        # Map utterances to their closest speaker to get new IUs
        delete_utterance = []
        for utt_key, utterance in self.utterances.items():
            speaker_keys = self.speaker_timeline.keys()
            for i, speaker_key in enumerate(speaker_keys):
                speaker = self.speaker_timeline[speaker_key]
                # Anything before first speaker is first speaker
                if i == 0 and utt_key <= speaker_key:
                    delete_utterance.append(utt_key)
                    self._add_new_utterance(um, speaker, utterance)
                    break
                # Anything after last speaker is unconfirmed as the last speaker
                elif i == len(speaker_keys) - 1 and utt_key >= speaker_key:
                    delete_utterance.append(utt_key)
                    self.last_text = utt_key
                    speaker_id, _ = speaker.get_speaker()
                    if speaker_id is not None:
                        output_iu = self.create_iu(speaker)
                        output_iu.set_speaker(speaker_id)
                        output_iu.set_text(utterance)
                        um.add_iu(output_iu, retico_core.UpdateType.ADD)
                        self.current_output.append(output_iu)
                    break
                # Otherwise check if closer to current or next speaker
                elif i != 0 and utt_key >= speaker_key and utt_key <= speaker_keys[i + 1]:
                    # Check which speaker the utterance is closer to by comparing the grounded audio input
                    if self._closer_to(utt_key, speaker_key, speaker_keys[i + 1]):
                        speaker = self.speaker_timeline[speaker_keys[i + 1]]
                    delete_utterance.append(utt_key)
                    self._add_new_utterance(um, speaker, utterance)

        for key in delete_utterance:
            del self.utterances[key]

    def process_update(self, update_message):
        um = retico_core.UpdateMessage()
        speakers_updated = []
        text_added = False
        old_last_speaker = self.speaker_timeline.keys()[len(
            self.speaker_timeline) - 1] if len(self.speaker_timeline) != 0 else None
        for i, tup in enumerate(update_message):
            iu, ut = tup
            origin = iu.grounded_in.created_at
            # Only deal with text commits
            if isinstance(iu, SpeechRecognitionIU) and ut == retico_core.UpdateType.COMMIT and i == len(update_message) - 1:
                self.utterances[origin] = iu.predictions[0]
                text_added = True
            elif isinstance(iu, SpeakerIU):
                # Update any speakers
                self.speaker_timeline[origin] = iu
                speakers_updated.append(origin)
        # Update existing ius if speakers changed
        if len(speakers_updated) > 0:
            self._update_existing_ius(um, speakers_updated, old_last_speaker)
        # Handle new text/existing text
        if len(speakers_updated) > 0 or text_added:
            self._create_new_utterances(um)
        # If any new ius append update message
        if len(um) > 0:
            self.append(um)
