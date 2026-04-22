# UI And Sentiment Accuracy Design

## Goal

Improve the web experience for login and multimodal sentiment analysis, and raise practical judgment accuracy by making text, video, language routing, transcript fusion, confidence, and warnings visible and usable.

## Recommended Approach

Use the existing Ant Design Pro stack and the current MSA service APIs. Repair the corrupted Chinese copy, redesign the login and analysis pages, and wire the existing `enhanceTextWithTranscript` option into the frontend so manual text can be fused with video ASR when a video is uploaded.

This avoids risky model retraining and uses capabilities already present in the service: dataset routing by language, condition-specific checkpoints, multimodal feature extraction, ASR transcript extraction, confidence, and warnings.

## User Experience

- Login presents a polished product-first screen with a clear brand, short value proposition, and a focused account form.
- Analysis opens directly into the workspace: text input, video upload, language selection, transcript enhancement toggle, submit action, progress, and final results.
- Results show polarity, confidence, score, used modalities, selected dataset, selected model condition, transcript, feature status, and warnings.
- Low-confidence or degraded-modality results should be visible enough that users do not over-trust uncertain predictions.

## Accuracy Strategy

- Preserve automatic language detection for text.
- Let users explicitly choose Chinese or English when detection is ambiguous.
- When both manual text and video are present, expose and default-enable transcript fusion so speech evidence can supplement user text.
- Keep result metadata visible so users can tell whether the model used text only, video only, or full audio-text-video inference.

## Testing

- Frontend build should fail before repair on the corrupted TSX and pass after the UI rewrite.
- Backend tests should continue to verify `enhanceTextWithTranscript` is forwarded only when text and video are present.
- MSA tests should continue to verify text+ASR fusion and model routing behavior.
