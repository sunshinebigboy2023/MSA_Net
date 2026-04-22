from .schemas import PredictionResult

__all__ = ["infer_runtime_from_checkpoint", "PredictionResult"]


def __getattr__(name):
    if name == "infer_runtime_from_checkpoint":
        from .checkpoint import infer_runtime_from_checkpoint

        return infer_runtime_from_checkpoint
    raise AttributeError(name)
