import unittest


class TaskManagerTests(unittest.TestCase):
    def test_create_task_starts_in_pending_state(self):
        from msa_service.dao.task_dao import InMemoryTaskDao

        manager = InMemoryTaskDao()
        task = manager.create({"text": "hello"})

        self.assertEqual(task.status, "PENDING")
        self.assertEqual(task.payload["text"], "hello")
