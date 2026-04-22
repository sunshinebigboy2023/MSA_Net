# MSA-Net High Concurrency Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a visible Redis + RabbitMQ + durable task workflow around MSA-Net inference.

**Architecture:** Java creates and persists analysis tasks, protects submissions with Redis rate limiting, publishes work to RabbitMQ, and accepts Python worker callbacks. Python consumes queue messages with bounded concurrency and reuses the existing `AnalysisService` for feature extraction and model inference.

**Tech Stack:** Spring Boot 2.6, Java 8, MyBatis-Plus, Redis, RabbitMQ, Python, pika, requests, existing MSA-Net service code.

---

### Task 1: Backend Task Model and SQL

**Files:**
- Create: `Net/user-center-backend-master/src/main/java/com/yupi/usercenter/model/domain/analysis/AnalysisTask.java`
- Create: `Net/user-center-backend-master/src/main/java/com/yupi/usercenter/mapper/AnalysisTaskMapper.java`
- Modify: `Net/user-center-backend-master/sql/create_table.sql`

**Steps:**
1. Write mapper/status unit tests around task status constants and response conversion.
2. Add `analysis_task` table with payload/result JSON, status, user id, retry count, timestamps, and error.
3. Add entity and mapper using MyBatis-Plus.

### Task 2: Redis Cache and Rate Limiting

**Files:**
- Create: `AnalysisCacheService.java`
- Create: `AnalysisRateLimitService.java`
- Modify: `MsaProperties.java`
- Test: focused unit tests with mocked Redis operations where practical.

**Steps:**
1. Test that cache keys and result/task serialization are stable.
2. Test rate limiter denies when Redis Lua returns 0.
3. Implement Redis-backed cache and Lua token-window limiter.

### Task 3: RabbitMQ Queue Layer

**Files:**
- Create: `RabbitMqConfig.java`
- Create: `AnalysisTaskMessage.java`
- Create: `AnalysisQueueProducer.java`
- Test: producer unit test with mocked `RabbitTemplate`.

**Steps:**
1. Test that producer sends the expected message to the configured exchange/routing key.
2. Declare work, retry, and dead-letter queues.
3. Implement publisher.

### Task 4: Async Analysis Submission and Callback

**Files:**
- Modify: `AnalysisServiceImpl.java`
- Create: `AnalysisTaskService.java`
- Create: `AnalysisCallbackController.java`
- Create request DTOs for callback payload.
- Test: service creates queued task and callback updates final result.

**Steps:**
1. Test async submit returns `QUEUED` and publishes exactly one queue message.
2. Test callback stores success result and cache.
3. Keep direct MSA client fallback for old tests.

### Task 5: Compensation Job

**Files:**
- Create: `AnalysisCompensationJob.java`

**Steps:**
1. Test stale queued/running tasks are requeued until max retries.
2. Mark tasks failed after retry budget.

### Task 6: Python Worker

**Files:**
- Create: `MSA/msa_service/worker.py`
- Create: `MSA/msa_service/service/callback_client.py`
- Modify: `MSA/requirements-standalone.txt`
- Test: unit test for bounded executor/callback payload.

**Steps:**
1. Test message payload is converted into an `AnalysisService` task and callback payload.
2. Implement RabbitMQ consumer using `pika`.
3. Run inference in a bounded `ThreadPoolExecutor`.
4. Ack on callback success, nack/requeue on recoverable failure.

### Task 7: Verification

**Commands:**
- `.\mvnw.cmd test "-Dtest=AnalysisServiceImplTest,AnalysisControllerTest,HttpMsaClientTest"`
- `python -m pytest MSA/tests/test_http_server.py MSA/tests/test_task_manager.py`

**Expected:** Existing tests pass, new focused tests pass without requiring live Redis/RabbitMQ.
