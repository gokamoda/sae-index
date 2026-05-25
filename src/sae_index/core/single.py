from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class Activations:
    prompt_id: int
    token_ids: NDArray[np.int64]  # shape: (sequence,)
    feature_ids: NDArray[np.int64]  # shape: (sequence, top_k)
    activations: NDArray[np.float32]  # shape: (sequence, top_k)

    def __post_init__(self):
        assert (
            self.token_ids.shape[0]
            == self.feature_ids.shape[0]
            == self.activations.shape[0]
        )
        assert self.feature_ids.shape[1] == self.activations.shape[1]
