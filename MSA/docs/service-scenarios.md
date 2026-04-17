# Standalone Service Input Scenarios

The service supports common local input scenarios. It never marks a modality as used unless a real feature vector was supplied to MoMKE.

## Checkpoint Routing

The service now discovers condition-specific checkpoints under `models/` by filename, for example `test-condition-t.pth`.

Supported model conditions:

- `a`: audio feature only.
- `t`: text feature only.
- `at`: audio + text features.
- `atv`: audio + text + video features.

The final JSON includes `modelCondition` so callers can verify which checkpoint was used. Unsupported feature combinations such as `v`, `av`, or `tv` fail explicitly unless a matching checkpoint is added.

## 1. Text Only

Input:

```json
{"text": "用户输入的文本"}
```

Behavior:

- Extract DeBERTa text feature.
- Run MoMKE with the `t` checkpoint.

## 2. Audio Only

Input:

```json
{"audioFile": "path/to/audio.wav"}
```

Behavior:

- Run Whisper ASR to obtain transcript.
- Extract DeBERTa text feature from transcript.
- Extract a 512-dimensional wav2vec-large audio feature.
- Run MoMKE with the `at` checkpoint.

## 3. Video Only With Audio

Input:

```json
{"videoFile": "path/to/video.mp4"}
```

Behavior:

- Use FFmpeg to extract a 16k mono wav into `temp/<taskId>/audio.wav`.
- Run Whisper ASR on extracted audio.
- Extract DeBERTa text feature from transcript.
- Extract a 512-dimensional wav2vec-large audio feature.
- Run OpenFace to create aligned face images, then extract a 1024-dimensional manet video feature.
- Run MoMKE with the `atv` checkpoint.
- If a feature extractor is unavailable, the remaining feature combination must still have a matching checkpoint.

## 4. Video Only Without Audio

Input:

```json
{"videoFile": "path/to/silent-video.mp4"}
```

Behavior:

- Detect that no audio stream exists.
- Skip ASR and wav2vec.
- Run OpenFace to create aligned face images, then extract a 1024-dimensional manet video feature.
- Current four-checkpoint setup does not include a `v` checkpoint, so video-only-without-audio fails explicitly after visual feature extraction unless a matching `test-condition-v.pth` is added.

## 5. Video With Manual Text

Input:

```json
{
  "videoFile": "path/to/video.mp4",
  "text": "用户手动修正或补充的文本"
}
```

Behavior:

- Manual text takes priority over ASR.
- Extract DeBERTa text feature from user text.
- Extract audio from video and extract a 512-dimensional wav2vec-large audio feature.
- Run OpenFace to create aligned face images, then extract a 1024-dimensional manet video feature.
- Run MoMKE with the `atv` checkpoint when audio is available.
- If raw audio is unavailable, the remaining `tv` combination currently needs a matching checkpoint before it can run.

## Current Local Dependency State

- Whisper transcription works with the local model under `tools/whisiper medium`.
- FFmpeg extraction works through the winget-installed binary.
- wav2vec-large audio extraction works through the built-in compatibility loader for `tools/wav2vec/wav2vec_large.pt`; `fairseq` is no longer a runtime dependency.
- Raw manet visual extraction works through `tools/openface/FeatureExtraction.exe` and `tools/manet/[02-08]-[21-19]-model_best-acc88.33.pth`.
