# MSA-Net ECS Deployment Design

## Goal

Deploy MSA-Net as a web service on a single Alibaba Cloud ECS instance using Docker Compose, with a single public HTTP entrypoint and automatic cleanup of transient inference artifacts.

## Constraints

- Use one ECS instance only.
- Keep MySQL, Redis, RabbitMQ, backend, worker, and frontend on the same host.
- Access the service by public IP first, without HTTPS.
- Avoid conflict with the existing `reminder` service that uses port `5019`.
- Delete transient audio, frame, OpenFace, and other intermediate inference files as soon as tasks finish.

## Chosen Approach

Use a single Docker Compose stack with a dedicated `frontend` container serving the built React app through Nginx on host port `5020`. Nginx proxies `/api` to the Spring Boot backend over the Docker network. Internal services stay private to the Compose network, except an optional remapped RabbitMQ management port for maintenance.

## Public Surface

- `http://<ecs-ip>:5020/` -> frontend Nginx
- `http://<ecs-ip>:5020/api/...` -> backend API through Nginx reverse proxy
- Optional: `http://<ecs-ip>:15673/` -> RabbitMQ management UI

## Internal Topology

- `frontend`
  - Builds static assets from `Net/user-center-frontend-master`
  - Serves built files with Nginx
  - Proxies `/api` to `backend:8080`
- `backend`
  - Spring Boot application
  - Uses MySQL, Redis, RabbitMQ over the Docker network
  - Stores uploaded user videos in a shared runtime volume
- `msa-worker`
  - Consumes async tasks from RabbitMQ
  - Reads checkpoints from mounted `MSA/models`
  - Reads feature-extraction tools from mounted `MSA/tools`
  - Calls backend callback endpoint over the Docker network
- `mysql`, `redis`, `rabbitmq`
  - Internal only
  - Persistent named Docker volumes

## Port Plan

- Keep host `5019` untouched for `reminder`
- Use host `5020` for MSA-Net frontend
- Do not expose host `8080`, `3306`, `6379`, or `5672`
- If RabbitMQ management is needed, remap it to host `15673`

## Artifact Cleanup Policy

Two cleanup layers are needed:

1. Worker-side transient inference cleanup
- Task-local temp audio extracted from uploaded video
- OpenFace aligned face directories
- Any per-task temp directories under `temp/`
- Cleanup must run in `finally` so both success and failure paths are covered

2. Backend-side uploaded media cleanup
- Uploaded original videos under `runtime/uploads`
- Keep them while a task may still be retried
- Delete them only when the task reaches a terminal state:
  - `SUCCESS`
  - `FAILED`
  - `DEAD_LETTER`
- Do not delete when task state is `QUEUED`, `RUNNING`, or `RETRYING`

## Why This Split

The backend owns uploaded originals and retry state. The worker owns derived temp files created during feature extraction. If the worker deletes uploaded originals on first failure, retries break. If the backend keeps temp artifacts forever, disk usage grows without bound. Splitting ownership by lifecycle keeps retries safe and cleanup deterministic.

## Expected Repo Changes

- `docker-compose.yml`
  - Add `frontend` service
  - Expose only `5020` publicly for web access
  - Restrict internal service ports
  - Remap RabbitMQ management to `15673` if retained
- `Net/user-center-frontend-master/docker/nginx.conf`
  - Add `/api` reverse proxy to backend
  - Keep SPA route fallback to `index.html`
- `MSA/msa_service/service/analysis_service.py`
  - Add task-scoped temp cleanup hooks
- `MSA/msa_service/service/video_feature_service.py`
  - Make temp output paths task-scoped instead of generic shared paths
- `Net/user-center-backend-master`
  - Add terminal-state upload cleanup after callback processing
- `README.md`
  - Add ECS deployment steps
  - Document ports, model mounts, and security group requirements

## Operational Notes

- ECS security group should open `5020/tcp`
- Open `15673/tcp` only if RabbitMQ management is actually needed
- Model and tool directories remain host-mounted because of size
- This design is intentionally HTTP-only for the first deployment; HTTPS and domain-based routing can be added later with the same frontend Nginx entrypoint
