# MSA-Net ECS Deployment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Deploy MSA-Net on a single Alibaba Cloud ECS instance behind one public HTTP entrypoint on port `5020`, while automatically cleaning transient inference artifacts and uploaded media when tasks reach terminal states.

**Architecture:** Add a production `frontend` container that serves the built UI through Nginx and proxies `/api` to the backend on the Docker network. Keep all stateful services on the same Compose stack, restrict public ports, clean worker temp files immediately after task execution, and delete uploaded originals only after backend task status becomes terminal.

**Tech Stack:** Docker Compose, Nginx, React/Umi, Spring Boot, Python worker, RabbitMQ, Redis, MySQL

---

### Task 1: Add failing Python tests for worker temp cleanup

**Files:**
- Modify: `MSA/tests/test_analysis_service_scenarios.py`

**Step 1: Write the failing test**

Add tests that create a task-local temp directory and assert it is deleted after `run_task`, including a failure-path case.

**Step 2: Run test to verify it fails**

Run: `cd MSA; python -m unittest tests.test_analysis_service_scenarios.AnalysisServiceScenarioTests -v`

Expected: cleanup assertions fail because temp directories still exist.

**Step 3: Write minimal implementation**

Add task-scoped cleanup in `AnalysisService.run_task` and task-scoped temp path helpers used by audio extraction and video feature extraction.

**Step 4: Run test to verify it passes**

Run: `cd MSA; python -m unittest tests.test_analysis_service_scenarios.AnalysisServiceScenarioTests -v`

Expected: PASS

### Task 2: Add failing Java tests for terminal-state upload cleanup

**Files:**
- Modify: `Net/user-center-backend-master/src/test/java/com/yupi/usercenter/service/impl/AnalysisServiceImplTest.java`
- Create: `Net/user-center-backend-master/src/test/java/com/yupi/usercenter/service/impl/AnalysisTaskServiceCleanupTest.java`

**Step 1: Write the failing test**

Add tests that:
- save a video upload and verify it exists
- simulate callback completion to `SUCCESS` and verify the referenced file is deleted
- simulate callback completion to `RETRYING` and verify the file is preserved

**Step 2: Run test to verify it fails**

Run: `cd Net\\user-center-backend-master; .\\mvnw.cmd test "-Dtest=AnalysisServiceImplTest,AnalysisTaskServiceCleanupTest"`

Expected: deletion assertions fail because backend does not clean uploads yet.

**Step 3: Write minimal implementation**

Teach backend task completion logic to parse payload JSON, detect `videoFile`, and remove it only when the final callback status is terminal.

**Step 4: Run test to verify it passes**

Run: `cd Net\\user-center-backend-master; .\\mvnw.cmd test "-Dtest=AnalysisServiceImplTest,AnalysisTaskServiceCleanupTest"`

Expected: PASS

### Task 3: Add production frontend gateway in Compose

**Files:**
- Modify: `docker-compose.yml`
- Modify: `Net/user-center-frontend-master/docker/nginx.conf`

**Step 1: Add runtime packaging test expectations**

Extend packaging tests to assert presence of `frontend:` service, `5020:80` mapping, and absence of public `8080`, `3306`, `6379`, and `5672` mappings.

**Step 2: Run test to verify it fails**

Run: `cd MSA; python -m unittest tests.test_runtime_packaging -v`

Expected: FAIL because compose still exposes old ports and has no frontend service.

**Step 3: Update compose and Nginx**

- Add multistage frontend build container
- Publish only `5020:80`
- Proxy `/api` to backend
- Keep RabbitMQ management optional on `15673:15672`

**Step 4: Run test to verify it passes**

Run: `cd MSA; python -m unittest tests.test_runtime_packaging -v`

Expected: PASS

### Task 4: Update ECS deployment docs

**Files:**
- Modify: `README.md`

**Step 1: Document ECS deployment flow**

Add concise steps for:
- preparing ECS
- placing `MSA/models` and `MSA/tools`
- opening security-group port `5020`
- starting with `docker compose up -d --build`
- visiting `http://<ecs-ip>:5020`

**Step 2: Document cleanup behavior**

Explain that transient worker files and uploaded originals are cleaned automatically after task terminal states.

### Task 5: Run targeted verification

**Files:**
- No code changes

**Step 1: Run Python tests**

Run: `cd MSA; python -m unittest tests.test_analysis_service_scenarios tests.test_runtime_packaging -v`

Expected: PASS

**Step 2: Run Java tests**

Run: `cd Net\\user-center-backend-master; .\\mvnw.cmd test "-Dtest=AnalysisServiceImplTest,AnalysisTaskServiceCleanupTest"`

Expected: PASS

**Step 3: Validate compose syntax**

Run: `docker compose config --quiet`

Expected: exit code `0`
