# MSA-Net High Concurrency Design

## Goal

Turn MSA-Net from a request-bound inference demo into an asynchronous, high-concurrency AI inference service suitable for production-style resume presentation.

## Architecture

The Java backend accepts analysis requests, validates and stores uploaded media, creates a durable `analysis_task` row, writes hot task metadata to Redis, and publishes a task event to RabbitMQ. Python workers consume task events with bounded concurrency, run the existing MSA-Net pipeline, and call back to Java with success or failure. Java persists the final result, refreshes Redis cache, and exposes polling APIs for task state and result.

## High-Concurrency Features

- Redis Lua user-level rate limiting to protect expensive inference resources.
- Redis result/task cache for frequent polling and repeated result reads.
- RabbitMQ exchange, work queue, retry queue, and dead-letter queue for traffic shaping and failure isolation.
- Durable MySQL task table so requests survive process restarts.
- Compensation job that scans stale queued/running tasks and re-publishes or marks them failed.
- Python worker pool that limits CPU/GPU concurrency instead of spawning one thread per request.
- Callback API for final consistency between Python inference and Java task state.

## Accuracy and Stability Features

- Preserve text + video transcript enhancement.
- Preserve language-aware routing metadata.
- Preserve modality quality warnings from Python service.
- Cache completed results for deterministic repeated reads and less queue pressure.

## Implementation Notes

- Use RabbitMQ rather than Pulsar because this backend is Spring Boot 2.6 on Java 8. RabbitMQ is compatible, simple to run locally, and still demonstrates MQ-based削峰填谷.
- Keep the existing direct `MsaClient` path available for unit tests and fallback construction, but enable the async path when queue services are injected by Spring.
- Avoid requiring live Redis/RabbitMQ/MySQL in unit tests by testing serialization, status transitions, queue payload creation, and fallback behavior with fakes/mocks.
