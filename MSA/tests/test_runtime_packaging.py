import unittest
from pathlib import Path


class RuntimePackagingTests(unittest.TestCase):
    def test_docker_compose_contains_high_concurrency_services(self):
        root = Path(__file__).resolve().parents[2]
        compose = (root / "docker-compose.yml").read_text(encoding="utf-8")

        for service_name in ("mysql:", "redis:", "rabbitmq:", "backend:", "msa-worker:", "frontend:"):
            self.assertIn(service_name, compose)
        self.assertIn("msa.analysis.queue", compose)
        self.assertIn('- "5020:80"', compose)
        self.assertIn('- "15673:15672"', compose)
        self.assertNotIn('- "8080:8080"', compose)
        self.assertNotIn('- "3306:3306"', compose)
        self.assertNotIn('- "6379:6379"', compose)
        self.assertNotIn('- "5672:5672"', compose)
        self.assertTrue((root / "MSA" / "Dockerfile.worker").exists())


if __name__ == "__main__":
    unittest.main()
