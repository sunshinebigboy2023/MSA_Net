from .dao.task_dao import InMemoryTaskDao

__all__ = ["infer_runtime_from_checkpoint", "MoMKEPredictor", "MoMKEPredictorRegistry", "InMemoryTaskDao"]


def __getattr__(name):
    if name == "infer_runtime_from_checkpoint":
        from .domain.checkpoint import infer_runtime_from_checkpoint

        return infer_runtime_from_checkpoint
    if name in {"MoMKEPredictor", "MoMKEPredictorRegistry"}:
        from .service.predictor_service import MoMKEPredictor, MoMKEPredictorRegistry

        return {"MoMKEPredictor": MoMKEPredictor, "MoMKEPredictorRegistry": MoMKEPredictorRegistry}[name]
    raise AttributeError(name)
