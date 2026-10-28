"""Tournament training, workload generation, statistical gates, and selection funnel."""

from src.model.tournament.trainer import (
    MultiTaskBatchSampler,
    TrainerConfig,
    TournamentTrainer,
    build_adamw_optimizer,
)

__all__ = [
    "MultiTaskBatchSampler",
    "TrainerConfig",
    "TournamentTrainer",
    "build_adamw_optimizer",
]
