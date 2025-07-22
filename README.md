# retico-speakerdiarization

Incremental speech diarization module for [ReTiCo](https://github.com/retico-team), speech diarization is the task of identifying which speech belongs to which speakers.

### Requirements

This module was built using `python=3.9`, install with pip:
```bash
pip install .
```
You will likely want `cuBLAS` and `cuDNN` for GPU execution (see [here](https://pytorch.org/get-started/locally/)).

If you run into issues importing speechbrain:
```bash
git clone https://github.com/speechbrain/speechbrain.git
cd speechbrain
pip install -r requirements.txt
pip install --editable .
```

### Loading Initial Embeddings

The module requires an audio recording (WAV) of each speaker. This is then used to generate some synthetic data for each speaker for a classifier. I strongly recommend using the included sample recorder to obtain these.

You will need a backend to support `torchaudio.io` if you are loading initial wav files:
```bash
conda install -c conda-forge 'ffmpeg<7'
```
### Explanation

This module works under three assumptions:
1. The number of speakers is known/specified before hand
2. There are exactly two speakers (open to a PR to make this model handle multiple speakers)
3. Each utterance (speech ending in silence) belongs to exactly one speaker

The module receives AudioIUs similar to ASR, it buffers audio from this input. When there is silence (an utterance has ended) the following steps are taken:
1. Buffered audio is collected and passed to a speechbrain encoder to obtain a speaker embedding
2. Using the trained classifier predicts the speaker
3. If the similarity is above a sceptical threshold, the embedding is added to the training data the model is retrained (if there enough new real embeddings) and the speaker ID is COMMITTED in an incremental unit
4. Otherwise, the training data is not updated but the embedding and it's associated speaker ID are ADDED in an incremental unit. When this happens all IUs that are in an ADD but not COMMIT state, are checked again, following the same steps as above.

### Interpreting output IUs

When setting credulous/sceptical thresholds consider the following
- SpeakerIUs that are committed should be considered a strong guarantee of the speaker ID
- SpeakerIUs that are added are quite likely to guarantee the speaker ID, but not enough that they should train the model, they can later be REVOKED

### Example
```python
import retico_core
from retico_core import *
from retico_core.text import SpeechRecognitionIU
from retico_core.debug import DebugModule
from retico_core.audio import MicrophoneModule
from retico_speakerdiarization.speaker_diarization import SpeakerDiarizationModule, SpeakerIU



mic = MicrophoneModule()
debug = DebugModule()
sd = SpeakerDiarizationModule(
    audio_path='audio', sceptical_threshold=0.4)

mic.subscribe(sd)
sd.subscribe(debug)

mic.run()
sd.run()
debug.run()

input('Press enter to ...')

debug.stop()
sd.stop()
mic.stop()

```
### Utterance module
Also included for convenience is an utterance module. This module takes in ASR and speaker incremental units, and matches them based on their timestamps. It then outputs an utterance incremental unit which contains the text and speaker for an entire utterance (an utterance ends when there is silence).
