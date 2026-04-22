# MSA-Net High-Concurrency Demo Guide

This guide turns the code changes into a repeatable project demo for resumes, interviews, and coursework defense.

## 1. What To Show

The demo should prove that MSA-Net is no longer a synchronous model-call demo. It is now an asynchronous inference platform:

```text
Frontend submit
-> Spring Boot API
-> Redis Lua rate limit
-> MySQL analysis_task row
-> RabbitMQ msa.analysis.queue
-> bounded Python worker
-> MSA-Net inference
-> Java callback
-> MySQL + Redis result cache
-> frontend polling
```

Key source files to keep open during a code walk:

- `Net/user-center-backend-master/src/main/java/com/yupi/usercenter/service/impl/AnalysisServiceImpl.java`
- `Net/user-center-backend-master/src/main/java/com/yupi/usercenter/config/RabbitMqConfig.java`
- `Net/user-center-backend-master/src/main/java/com/yupi/usercenter/service/impl/AnalysisRateLimitService.java`
- `Net/user-center-backend-master/src/main/java/com/yupi/usercenter/service/impl/AnalysisTaskService.java`
- `Net/user-center-backend-master/src/main/java/com/yupi/usercenter/job/AnalysisCompensationJob.java`
- `MSA/msa_service/worker.py`
- `docker-compose.yml`

## 2. Prepare Local Assets

Large model files are intentionally ignored by Git. Before running a real inference demo, place them here:

```text
MSA/models/
MSA/tools/
```

The worker reads `MSA_CHECKPOINT=/app/models` in Docker Compose. If the checkpoint folder is empty, the architecture can still be reviewed, but real inference will fail and tasks may enter retry/dead-letter flow.

## 3. Start The Full Runtime

From the repository root:

```powershell
docker compose up --build
```

Expected services:

- Backend API: `http://localhost:8080/api`
- RabbitMQ console: `http://localhost:15672`
- RabbitMQ login: `msa` / `msa_password`
- Redis: `localhost:6379`
- MySQL: `localhost:3306`, database `yupi`
- Worker queue: `msa.analysis.queue`

If Docker is not installed on the machine, explain the architecture with the source files and run the unit tests below.

## 4. Submit A Task

Use the frontend if it is running, or call the API with a logged-in browser session.

Frontend route:

```text
http://localhost:8001/
```

The visible state flow should be:

```text
QUEUED -> RUNNING -> SUCCESS
```

If the model files are missing or invalid, the failure flow is still useful for explaining reliability:

```text
QUEUED -> RUNNING -> RETRYING -> DEAD_LETTER
```

## 5. Inspect RabbitMQ

Open:

```text
http://localhost:15672
```

Show:

- `msa.analysis.exchange`
- `msa.analysis.queue`
- `msa.analysis.retry.queue`
- `msa.analysis.dlq`
- Queue publish/consume rates
- Message count changes while submitting tasks

Talking point:

> RabbitMQ absorbs traffic spikes. The Java request thread only creates a task and publishes a message; slow model inference is handled by worker consumers.

## 6. Inspect MySQL

Use any MySQL client:

```sql
use yupi;

select taskId, userId, status, retryCount, processingTimeMs, createTime, updateTime
from analysis_task
order by id desc
limit 20;
```

For a single task:

```sql
select taskId, status, payload, result, error
from analysis_task
where taskId = 'replace-with-task-id';
```

Talking point:

> MySQL is the durable source of truth. Redis is used for hot reads and fast protection, but task state is not lost if the backend restarts.

## 7. Inspect Redis

With `redis-cli`:

```bash
keys msa:rate:submit:*
keys msa:task:*
keys msa:result:*
```

Talking point:

> Redis protects the expensive inference service with Lua-based rate limiting and speeds up high-frequency polling.

## 8. Run A Lightweight Load Test

First log in through the frontend and copy the `JSESSIONID` cookie. Then run:

```powershell
python scripts/load_test_analysis.py --total 100 --concurrency 20 --cookie "JSESSIONID=replace-with-session"
```

Useful variants:

```powershell
python scripts/load_test_analysis.py --total 300 --concurrency 50 --cookie "JSESSIONID=replace-with-session"
python scripts/load_test_analysis.py --total 100 --concurrency 20 --language zh --text "这个系统的异步推理效果很好" --cookie "JSESSIONID=replace-with-session"
```

Metrics to screenshot:

- `successRate`
- `avgMs`
- `p95Ms`
- `qps`
- `statusCounts`

Important explanation:

> This script measures submission throughput, not model completion throughput. That is intentional: high concurrency is handled by quickly accepting and queueing tasks while workers process inference at a controlled rate.

## 9. Screenshot Checklist

Capture these for a resume project album or defense slides:

- Frontend task submitted and showing `QUEUED`.
- Frontend task showing `RUNNING` or final result.
- RabbitMQ queue page showing `msa.analysis.queue`.
- MySQL `analysis_task` rows with changing statuses.
- Redis keys for `msa:task:*` or `msa:rate:submit:*`.
- Load test JSON output.
- Source code tabs: `RabbitMqConfig`, `AnalysisRateLimitService`, `worker.py`.

## 10. Common Questions

### Why not call Python directly from Java?

Direct calls block request threads and make traffic spikes hit the model service immediately. The queue decouples request acceptance from expensive inference.

### Why still keep the old HTTP MSA service?

It is useful for local debugging and fallback tests. The high-concurrency production path is RabbitMQ plus Python worker.

### Why RabbitMQ instead of Pulsar?

The reference `yu-like-main` project uses Pulsar, but this backend is Spring Boot 2.6 and Java 8. RabbitMQ is simpler to integrate with this stack while preserving the same high-concurrency idea: queue-based削峰填谷, retry, and dead-letter isolation.

### Why bounded worker concurrency?

Model inference is CPU/GPU and memory heavy. Unlimited threads can make latency worse or crash the process. A bounded worker pool protects inference resources.

### What improves accuracy?

The accuracy-oriented parts are text-video transcript fusion, language-aware routing, modality quality warnings, and avoiding bad modality features when extraction fails.

## 11. Commands To Verify Code Without Docker

Backend:

```powershell
cd Net/user-center-backend-master
.\mvnw.cmd test "-Dtest=AnalysisControllerTest,AnalysisServiceImplTest,HttpMsaClientTest,MsaPropertiesTest,AnalysisQueueProducerTest,AnalysisServiceImplAsyncTest"
```

Python:

```powershell
cd MSA
python -m unittest tests.test_runtime_packaging tests.test_worker_payload tests.test_http_server tests.test_load_test_analysis -v
```

Frontend:

```powershell
cd Net/user-center-frontend-master
npm.cmd test -- --runTestsByPath src/pages/analysisTaskStatus.test.ts
npm.cmd run build
```

