# MSA Net Integration Design

## Goal

Connect the existing MSA multimodal sentiment analysis service with the Net frontend and backend so logged-in users can analyze text, video, or both from a web page.

## Architecture

Use the Spring Boot backend as the integration boundary. The frontend submits text and optional video files to the Java backend. The backend keeps the existing login/session behavior, stores uploaded videos in a runtime upload directory, and calls the local Python MSA HTTP service at `http://127.0.0.1:8000`.

The MSA service remains an independent Python process. It continues to own model loading, feature extraction, inference, task status, and result formatting.

## User Flow

After login, the user lands on the main analysis page. The page allows:

- Optional text input.
- Optional video upload.
- Language selection: auto, Chinese, or English.
- Submission and live task progress.
- Result display with sentiment, score, confidence, used modalities, detected language, selected model dataset, model condition, transcript, and processing time.

Registration and login remain available under the existing user routes.

## Backend Flow

The Java backend exposes:

- `POST /api/analysis/analyze`
- `GET /api/analysis/task/{taskId}`
- `GET /api/analysis/result/{taskId}`

`POST /api/analysis/analyze` accepts `multipart/form-data`. It validates that at least text or video is present. If a video exists, it saves the file under a runtime upload directory such as `runtime/uploads/<userId>/<uuid>.<ext>`.

The backend forwards a JSON request to MSA:

```json
{
  "text": "optional text",
  "videoFile": "absolute saved video path",
  "language": "zh or en when the user overrides auto"
}
```

The backend wraps MSA responses in the existing `BaseResponse` format so the current frontend request interceptor can keep working.

## Frontend Flow

Replace the current default welcome page with a focused analysis workspace. Keep Ant Design Pro layout and login guard.

The page includes:

- A text area for typed sentiment content.
- A video upload control for a single media file.
- A segmented language selector.
- A submit button with loading state.
- A progress/status area while MSA runs.
- A structured result area after success.
- Clear failure messages when MSA is unavailable, input is invalid, or the task fails.

The route `/welcome` remains the main route, and `/` continues to redirect to `/welcome`.

## Configuration

Add backend properties for:

- MSA base URL, default `http://127.0.0.1:8000`.
- Upload directory, default `runtime/uploads`.
- Maximum upload size, configured through Spring multipart settings.

No model files, datasets, temporary outputs, or uploaded media should be committed.

## Error Handling

Invalid input returns a parameter error. MSA connection failures return a system error with a user-readable description. Task failures return the MSA task error in a controlled response. The frontend renders these states without losing the current form input.

## Testing

Backend tests use a fake `MsaClient` or injectable HTTP boundary to verify:

- Text-only submissions forward text.
- Video submissions save a file and forward the saved path.
- Empty submissions fail validation.
- Task and result proxy endpoints return MSA data.
- MSA failures become friendly backend errors.

Frontend checks cover:

- API wrapper shapes.
- Submit button disabled state for empty input.
- Correct multipart payload construction.
- Result rendering for a representative MSA response.

Implementation follows TDD: write failing tests first, verify they fail, then implement the minimal code.
