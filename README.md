# retico-speakerdiarization

Incremental speech diarization module for [ReTiCo](https://github.com/retico-team), speech diarization is the task of identifying which speech belongs to which speakers.

### Requirements

This module was built using `python=3.9`, install with pip:
```
pip install .
```
You will likely want `cuBLAS` and `cuDNN` for GPU execution (see [here](https://pytorch.org/get-started/locally/)).

### Explanation

This module works under a two assumptions:
1. The number of speakers is known/specified before hand
2. Each utterance (speech ending in silence) belongs to exactly one speaker

The module receives AudioIUs similar to ASR, it buffers audio from this input. When there is silence (an utterance has ended) the following steps are taken:
1. Buffered audio is collected and passed to a speechbrain encoder to obtain a speaker embedding
2. The embedding is compared to an existing embedding for each speaker (centroid) using cosine similarity
3. If the similarity is above a sceptical threshold, the centroid (average) is updated using the new embedding, and the speaker ID is COMMITTED in an incremental unit
4. If the similarity is not above a sceptical threshold, but is above a credulous threshold, the centroid is not updated but the embedding and it's associated speaker ID are ADDED in an incremental unit
5. Otherwise, if there is still room for more speakers the embedding becomes a new centroid, if there is not more room the embedding is discarded
Whenever, a new embedding is committed the centroids are updated. When this happens all IUs that are in an ADD but not COMMIT state, are checked again, following the same steps as above.

### Loading Initial Centroids

Normally the module starts with no centroids. This can be irritating, as if the microphone picks up background noise when there are speaker slots left it will make that embedding a centroid. As an alternative you can specify `audio_path` to point to a directory containing audio files. These audio files should be named after each speaker. These are then processed at startup. The initial embeddings will become the initial centroids, and the speaker IDs will be derived from the filenames. A utility script for recording 10 second clips of audio is included for convenience.

You will need a backend to support `torchaudio.io` if you are loading initial wav files:
```
conda install -c conda-forge 'ffmpeg<7'
```

### Interpreting output IUs

When setting credulous/sceptical thresholds consider the following
- SpeakerIUs that are committed should be considered a strong guarantee of the speaker ID
- SpeakerIUs that are added are quite likely to guarantee the speaker ID, but not enough that they should update centroids, they can later be REVOKED

### Example
```
import retico_core
from retico_core import *
from retico_core.text import SpeechRecognitionIU
from retico_core.debug import DebugModule
from retico_core.audio import MicrophoneModule
from retico_speakerdiarization.speaker_diarization import SpeakerDiarizationModule, SpeakerIU



mic = MicrophoneModule()
debug = DebugModule()
sd = SpeakerDiarizationModule(
    audio_path='audio', sceptical_threshold=0.6, credulous_threshold=0.3)

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
