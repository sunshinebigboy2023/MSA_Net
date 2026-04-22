import unittest
from pathlib import Path


class RuntimePackagingTests(unittest.TestCase):
    def test_docker_compose_contains_high_concurrency_services(self):
        root = Path(__file__).resolve().parents[2]
        compose = (root / "docker-compose.yml").read_text(encoding="utf-8")

        for service_name in ("mysql:", "redis:", "rabbitmq:", "backend:", "msa-worker:"):
            self.assertIn(service_name, compose)
        self.assertIn("msa.analysis.queue", compose)
        self.assertTrue((root / "MSA" / "Dockerfile.worker").exists())


if __name__ == "__main__":
    unittest.main()
