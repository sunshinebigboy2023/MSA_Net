from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from msa_service.service.analysis_service import AnalysisService
    from msa_service.service.callback_client import CallbackClient


STATUS_SUCCESS = "SUCCESS"
STATUS_FAILED = "FAILED"
STATUS_RUNNING = "RUNNING"

ANALYSIS_EXCHANGE = "msa.analysis.exchange"
ANALYSIS_DLX = "msa.analysis.dlx"
ANALYSIS_ROUTING_KEY = "msa.analysis.dispatch"
ANALYSIS_DLQ = "msa.analysis.dlq"
ANALYSIS_DLQ_ROUTING_KEY = "msa.analysis.dead"


def build_running_callback_payload(task_id: str) -> Dict[str, Any]:
    return {
        "taskId": task_id,
        "status": STATUS_RUNNING,
        "result": None,
        "error": None,
        "processingTimeMs": None,
    }


def build_success_callback_payload(task_id: str, result: Dict[str, Any], processing_time_ms: int) -> Dict[str, Any]:
    return {
        "taskId": task_id,
        "status": STATUS_SUCCESS,
        "result": result,
        "error": None,
        "processingTimeMs": processing_time_ms,
    }


def build_failure_callback_payload(task_id: str, error: str, processing_time_ms: int) -> Dict[str, Any]:
    return {
        "taskId": task_id,
        "status": STATUS_FAILED,
        "result": None,
        "error": error,
        "processingTimeMs": processing_time_ms,
    }


class AnalysisWorker:
    def __init__(self, service: "AnalysisService", callback_client: "CallbackClient"):
        self.service = service
        self.callback_client = callback_client

    def handle(self, message: Dict[str, Any]) -> None:
        external_task_id = str(message["taskId"])
        payload = dict(message.get("payload") or {})
        started_at = time.perf_counter()
        try:
            self.callback_client.complete(build_running_callback_payload(external_task_id))
            local_task = self.service.submit(payload)
            result = self.service.run_task(local_task.task_id)
            elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            self.callback_client.complete(build_success_callback_payload(external_task_id, result, elapsed_ms))
        except Exception as exc:
            elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            self.callback_client.complete(build_failure_callback_payload(external_task_id, str(exc), elapsed_ms))
            raise


def run_consumer(args) -> None:
    import pika
    from msa_service.service.analysis_service import AnalysisService
    from msa_service.service.callback_client import CallbackClient

    service = AnalysisService(args.checkpoint)
    callback_client = CallbackClient(args.callback_base_url, args.callback_token)
    worker = AnalysisWorker(service, callback_client)
    executor = ThreadPoolExecutor(max_workers=args.concurrency)

    credentials = pika.PlainCredentials(args.rabbitmq_username, args.rabbitmq_password)
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(
            host=args.rabbitmq_host,
            port=args.rabbitmq_port,
            virtual_host=args.rabbitmq_virtual_host,
            credentials=credentials,
            heartbeat=60,
        )
    )
    channel = connection.channel()
    channel.exchange_declare(exchange=ANALYSIS_EXCHANGE, exchange_type="direct", durable=True)
    channel.exchange_declare(exchange=ANALYSIS_DLX, exchange_type="direct", durable=True)
    channel.queue_declare(
        queue=args.queue,
        durable=True,
        arguments={
            "x-dead-letter-exchange": ANALYSIS_DLX,
            "x-dead-letter-routing-key": ANALYSIS_DLQ_ROUTING_KEY,
        },
    )
    channel.queue_bind(queue=args.queue, exchange=ANALYSIS_EXCHANGE, routing_key=ANALYSIS_ROUTING_KEY)
    channel.queue_declare(queue=ANALYSIS_DLQ, durable=True)
    channel.queue_bind(queue=ANALYSIS_DLQ, exchange=ANALYSIS_DLX, routing_key=ANALYSIS_DLQ_ROUTING_KEY)
    channel.basic_qos(prefetch_count=args.concurrency)

    def on_message(ch, method, _properties, body):
        delivery_tag = method.delivery_tag

        def execute_and_ack():
            try:
                message = json.loads(body.decode("utf-8"))
                worker.handle(message)
                connection.add_callback_threadsafe(lambda: ch.basic_ack(delivery_tag=delivery_tag))
            except Exception:
                connection.add_callback_threadsafe(lambda: ch.basic_nack(delivery_tag=delivery_tag, requeue=False))

        executor.submit(execute_and_ack)

    channel.basic_consume(queue=args.queue, on_message_callback=on_message, auto_ack=False)
    channel.start_consuming()


def parse_args():
    parser = argparse.ArgumentParser(description="RabbitMQ worker for MSA-Net analysis tasks")
    parser.add_argument("--checkpoint", default=os.environ.get("MSA_CHECKPOINT"))
    parser.add_argument("--rabbitmq-host", default=os.environ.get("RABBITMQ_HOST", "127.0.0.1"))
    parser.add_argument("--rabbitmq-port", type=int, default=int(os.environ.get("RABBITMQ_PORT", "5672")))
    parser.add_argument("--rabbitmq-username", default=os.environ.get("RABBITMQ_USERNAME", "guest"))
    parser.add_argument("--rabbitmq-password", default=os.environ.get("RABBITMQ_PASSWORD", "guest"))
    parser.add_argument("--rabbitmq-virtual-host", default=os.environ.get("RABBITMQ_VHOST", "/"))
    parser.add_argument("--queue", default=os.environ.get("MSA_ANALYSIS_QUEUE", "msa.analysis.queue"))
    parser.add_argument("--concurrency", type=int, default=int(os.environ.get("MSA_WORKER_CONCURRENCY", "2")))
    parser.add_argument("--callback-base-url", default=os.environ.get("MSA_CALLBACK_BASE_URL", "http://127.0.0.1:8080/api"))
    parser.add_argument("--callback-token", default=os.environ.get("MSA_CALLBACK_TOKEN", "msa-worker-token"))
    return parser.parse_args()


def main() -> None:
    run_consumer(parse_args())


if __name__ == "__main__":
    main()
