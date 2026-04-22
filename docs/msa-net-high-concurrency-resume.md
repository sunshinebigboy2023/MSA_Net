# MSA-Net High-Concurrency Resume Notes

## Project Summary

MSA-Net is a multimodal sentiment analysis platform that supports text and video inputs. I refactored the original synchronous inference call into an asynchronous high-concurrency architecture using Redis, RabbitMQ, MySQL, and a bounded Python worker pool.

## Resume Bullets

- Refactored synchronous multimodal inference into a Redis + RabbitMQ async task pipeline, reducing request thread blocking and improving traffic burst tolerance.
- Designed a durable `analysis_task` state machine in MySQL with `QUEUED`, `RUNNING`, `SUCCESS`, `FAILED`, `RETRYING`, and `DEAD_LETTER` states.
- Implemented Redis Lua based user-level submit rate limiting and Redis caching for hot task/result polling.
- Built RabbitMQ work, retry, and dead-letter queues to support retry isolation and failure diagnosis.
- Implemented a Python MSA-Net worker with bounded concurrency to protect CPU/GPU inference resources.
- Added worker callback and compensation job to keep Java task state eventually consistent with Python inference results.
- Added Docker Compose packaging for MySQL, Redis, RabbitMQ, backend, and Python worker, plus a lightweight concurrency test script.

## Interview Talking Points

- Why MQ: model inference is slow and resource-heavy, so the request path should only validate input, persist metadata, and enqueue work.
- Why Redis: high-frequency polling and repeated submit checks should not always hit MySQL; Redis also provides fast atomic rate limiting.
- Why bounded worker concurrency: unlike normal CRUD, AI inference can exhaust CPU/GPU memory if every request starts a new thread.
- Why dead-letter queue: failed inference tasks need to be isolated for diagnosis instead of being retried forever.
- Why compensation job: callbacks and consumers may fail, so stale tasks are scanned and requeued or marked dead-letter.

## Demo Script

1. Start infrastructure with `docker compose up --build`.
2. Open RabbitMQ console at `http://localhost:15672`.
3. Submit several analysis tasks from the frontend.
4. Show task status moving through `QUEUED` and `RUNNING`.
5. Show RabbitMQ queue metrics and MySQL `analysis_task` rows.
6. Run `python scripts/load_test_analysis.py --total 100 --concurrency 20 --cookie "JSESSIONID=..."`.
