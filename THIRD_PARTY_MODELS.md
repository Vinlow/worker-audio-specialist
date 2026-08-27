# Third-party model artifacts

This container redistributes the exact model artifacts listed below for
offline inference. Web2Labs does not claim ownership of them. Unless an entry
says otherwise, Web2Labs made no changes to the upstream model or configuration
files; it only packaged them into this container. The Whisper CTranslate2
conversions were performed and published upstream.

## MIT-licensed artifacts

### Whisper and faster-whisper conversions

Original Whisper models: OpenAI, Copyright (c) 2022 OpenAI. CTranslate2
conversion tooling and models: SYSTRAN, Copyright (c) 2023 SYSTRAN.

- `Systran/faster-whisper-base`
  - Revision: `ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66`
  - Source: https://huggingface.co/Systran/faster-whisper-base/tree/ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66
- `Systran/faster-whisper-small`
  - Revision: `536b0662742c02347bc0e980a01041f333bce120`
  - Source: https://huggingface.co/Systran/faster-whisper-small/tree/536b0662742c02347bc0e980a01041f333bce120
- `Systran/faster-whisper-medium`
  - Revision: `08e178d48790749d25932bbc082711ddcfdfbc4f`
  - Source: https://huggingface.co/Systran/faster-whisper-medium/tree/08e178d48790749d25932bbc082711ddcfdfbc4f
- `Systran/faster-whisper-large-v3`
  - Revision: `edaa852ec7e145841d8ffdb056a99866b5f0a478`
  - Source: https://huggingface.co/Systran/faster-whisper-large-v3/tree/edaa852ec7e145841d8ffdb056a99866b5f0a478
- `mobiuslabsgmbh/faster-whisper-large-v3-turbo`
  - Canonical repository: `dropbox-dash/faster-whisper-large-v3-turbo`
  - Revision: `0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf`
  - Source: https://huggingface.co/dropbox-dash/faster-whisper-large-v3-turbo/tree/0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf

License sources:

- https://github.com/openai/whisper/blob/main/LICENSE
- https://github.com/SYSTRAN/faster-whisper/blob/v1.2.1/LICENSE

### Segment Any Text

`segment-any-text/sat-3l-sm`, by Benjamin Minixhofer, Markus Frohmann, and
Igor Sterner.

- Revision: `137da054051ad9f1eac42025f758db4ac9f22535`
- Source: https://huggingface.co/segment-any-text/sat-3l-sm/tree/137da054051ad9f1eac42025f758db4ac9f22535
- License source: https://github.com/segment-any-text/wtpsplit/blob/main/LICENSE

### XLM-RoBERTa tokenizer

`FacebookAI/xlm-roberta-base`, from Facebook AI/fairseq. Copyright (c)
Facebook, Inc. and its affiliates.

- Revision: `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`
- Source: https://huggingface.co/FacebookAI/xlm-roberta-base/tree/e73636d4f797dec63c3081bb6ed5c7b0bb3f2089
- License source: https://github.com/facebookresearch/fairseq/blob/main/LICENSE

### pyannote diarization

Pipeline and model authors include Alexis Plaquet and Hervé Bredin; pyannote
artifacts are published by pyannote/CNRS.

- `pyannote/speaker-diarization-3.1`
  - Revision: `84fd25912480287da0247647c3d2b4853cb3ee5d`
  - Source: https://huggingface.co/pyannote/speaker-diarization-3.1/tree/84fd25912480287da0247647c3d2b4853cb3ee5d
- `pyannote/segmentation-3.0`
  - Revision: `e66f3d3b9eb0873085418a7b813d3b369bf160bb`
  - Copyright (c) 2023 CNRS
  - Source: https://huggingface.co/pyannote/segmentation-3.0/tree/e66f3d3b9eb0873085418a7b813d3b369bf160bb

### wav2vec 2.0 alignment checkpoint

`torchaudio/WAV2VEC2_ASR_LARGE_LV60K_960H`, originally published by the
wav2vec 2.0/fairseq authors and redistributed by torchaudio under MIT.

- File: `wav2vec2_fairseq_large_lv60k_asr_ls960.pth`
- SHA-256: `7a88965716fbd598a595209bf45c1210a18a6935cfb0cf53527fc986c5543ac7`
- Source and license declaration: https://github.com/pytorch/audio/blob/v2.7.1/src/torchaudio/pipelines/_wav2vec2/impl.py

### MIT terms

Permission is hereby granted, free of charge, to any person obtaining a copy of
the software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The copyright notices above and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Apache-2.0 artifact

### LAION CLAP

`laion/larger_clap_music_and_speech`, published by LAION. Model authors include
Yusong Wu, Ke Chen, Tianyu Zhang, Yuchen Hui, Taylor Berg-Kirkpatrick, and
Shlomo Dubnov.

- Revision: `195c3a3e68faebb3e2088b9a79e79b43ddbda76b`
- Source: https://huggingface.co/laion/larger_clap_music_and_speech/tree/195c3a3e68faebb3e2088b9a79e79b43ddbda76b
- License: Apache License 2.0
- Changes: no model-file changes; packaged for offline inference.

The complete Apache License 2.0 is included in this container at
`/usr/share/licenses/audio-worker/Apache-2.0.txt`. The exact model snapshot
supplies no upstream `NOTICE` file.

## CC-BY-4.0 artifacts

### NVIDIA Parakeet

Title: `nvidia/parakeet-tdt-0.6b-v3`. Creator and publisher: NVIDIA.

- Revision: `7c35754d166cca382ad1e53e68b01e7c575f3a1d`
- Source: https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/tree/7c35754d166cca382ad1e53e68b01e7c575f3a1d
- License: https://creativecommons.org/licenses/by/4.0/legalcode.en
- Changes: no model-file changes; packaged for offline inference.

### WeSpeaker VoxCeleb speaker embedding

Title: `pyannote/wespeaker-voxceleb-resnet34-LM`. Creator: WeSpeaker authors;
pyannote wrapper by pyannote/CNRS.

- Revision: `837717ddb9ff5507820346191109dc79c958d614`
- Source: https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM/tree/837717ddb9ff5507820346191109dc79c958d614
- License: https://creativecommons.org/licenses/by/4.0/legalcode.en
- Changes: no weight or configuration changes; packaged for offline inference.

No endorsement by any upstream creator or publisher is implied.
