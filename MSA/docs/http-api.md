# Local HTTP API

Start the local service:

```powershell
python -m msa_service.controller.http_server --host 127.0.0.1 --port 8000
```

## POST /analyze

Submits one local analysis task. The service accepts raw inputs and precomputed feature paths.

The service automatically routes to condition-specific checkpoints discovered under `models/`:

- `a`: audio only.
- `t`: text only.
- `v`: video only.
- `at`: audio + text.
- `av`: audio + video.
- `tv`: text + video.
- `atv`: audio + text + video.

The service also detects text language before inference:

- Chinese text/transcript routes to `SIMS`.
- English or non-Chinese text routes to `CMUMOSI`.
- Inputs without text evidence default to `CMUMOSI`, unless the request includes `"language":"zh"` or `"language":"en"`.

The response result includes `language`, `modelDataset`, and `modelCondition` so callers can confirm which checkpoint was used.

Text only:

```powershell
curl -X POST http://127.0.0.1:8000/analyze `
  -H "Content-Type: application/json" `
  -d "{\"text\":\"我很高兴\"}"
```

English text:

```powershell
curl -X POST http://127.0.0.1:8000/analyze `
  -H "Content-Type: application/json" `
  -d "{\"text\":\"I am happy\"}"
```

Video only:

```powershell
curl -X POST http://127.0.0.1:8000/analyze `
  -H "Content-Type: application/json" `
  -d "{\"videoFile\":\"dataset/sims/0001.mp4\"}"
```

Video with manual text:

```powershell
curl -X POST http://127.0.0.1:8000/analyze `
  -H "Content-Type: application/json" `
  -d "{\"videoFile\":\"dataset/SIMS/Raw/video_0001/0001.mp4\",\"text\":\"我不想嫁给李茶\"}"
```

Force Chinese model when no text is available:

```powershell
curl -X POST http://127.0.0.1:8000/analyze `
  -H "Content-Type: application/json" `
  -d "{\"videoFile\":\"dataset/SIMS/Raw/video_0001/0001.mp4\",\"language\":\"zh\"}"
```

Audio only:

```powershell
curl -X POST http://127.0.0.1:8000/analyze `
  -H "Content-Type: application/json" `
  -d "{\"audioFile\":\"temp/sims_smoke/audio.wav\"}"
```

Precomputed features:

```powershell
curl -X POST http://127.0.0.1:8000/analyze `
  -H "Content-Type: application/json" `
  -d "{\"audioFeaturePath\":\"dataset/CMUMOSI/features/wav2vec-large-c-UTT/03bSnISJMiM_1.npy\",\"textFeaturePath\":\"dataset/CMUMOSI/features/deberta-large-4-UTT/03bSnISJMiM_1.npy\"}"
```

Response:

```json
{
  "taskId": "uuid",
  "status": "PENDING"
}
```

## GET /task/{id}

Returns current task status:

```powershell
curl http://127.0.0.1:8000/task/<taskId>
```

Statuses:

- `PENDING`
- `PREPROCESSING`
- `EXTRACTING`
- `INFERRING`
- `SUCCESS`
- `FAILED`

## GET /result/{id}

Returns the final structured result:

```powershell
curl http://127.0.0.1:8000/result/<taskId>
```

Result fields include:

- `taskId`
- `usedModalities`
- `missingModalities`
- `emotionLabel`
- `sentimentPolarity`
- `score`
- `confidence`
- `message`
- `error`
- `transcript`
- `featureStatus`
- `language`
- `modelDataset`
- `modelCondition`
- `rawInputs`
- `processingTimeMs`
