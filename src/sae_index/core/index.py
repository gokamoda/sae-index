from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .single import Activations


@dataclass
class ActivationExample:
    token_ids: NDArray[np.uint64]
    target_token_position: int
    activation_values: NDArray[np.float32]
    prompt_id: int | None = None


@dataclass
class Prompt:
    prompt_id: int
    query_token_position: int
    token_ids: NDArray[np.uint64]


@dataclass
class TopKActivations:
    feature_id: int
    global_token_positions: NDArray[np.uint64]
    activation_values: NDArray[np.float16]


class SAEActivationIndex:
    prompt_ids: NDArray[np.uint64]  # (num_prompts,)
    token_ids: NDArray[np.uint64]  # (total_tokens,)
    prompt_token_offsets: NDArray[np.uint64]  # (num_prompts + 1,)
    token_positions: NDArray[np.uint64]  # (nnz,)
    feature_offsets: NDArray[np.uint64]  # (hidden_size + 1,)
    activation_values: NDArray[np.float16]  # (nnz,)

    def __init__(
        self,
        activations: list[Activations],
        hidden_size: int,
    ):
        self.__init_token_index(activations)
        self.__init_feature_index(activations, hidden_size=hidden_size)

    def __init_token_index(
        self,
        activations: list[Activations],
    ):
        self.prompt_ids = np.asarray(
            [act.prompt_id for act in activations], dtype=np.uint64
        )

        self.token_ids = np.concatenate(
            [act.token_ids.astype(np.uint64, copy=False) for act in activations],
            axis=0,
        )

        lengths = np.asarray(
            [len(act.token_ids) for act in activations], dtype=np.uint64
        )
        prompt_token_offsets = np.zeros(
            len(activations) + 1, dtype=np.uint64
        )  # +1 to retrieve end position
        prompt_token_offsets[1:] = np.cumsum(lengths)
        self.prompt_token_offsets = prompt_token_offsets

    def __init_feature_index(
        self,
        activations: list[Activations],
        hidden_size: int,
    ):
        token_positions_flat, feature_ids_flat, values_flat = (
            self.__init_feature_index_phase1(activations)
        )
        token_positions, feature_offsets, activation_values = (
            self.__init_feature_index_phase2(
                token_positions_flat=token_positions_flat,
                feature_ids_flat=feature_ids_flat,
                values_flat=values_flat,
                hidden_size=hidden_size,
            )
        )
        self.token_positions = token_positions
        self.feature_offsets = feature_offsets
        self.activation_values = activation_values

    def __init_feature_index_phase1(
        self,
        activations: list[Activations],
    ):
        """
        prepare three flat arrays
        """
        all_token_positions = []
        all_feature_ids = []
        all_values = []

        global_position = 0

        for act in activations:
            sequence_length, feature_indices = act.feature_ids.shape

            # make something like: [0, 0, 0, ..., 1, 1, 1, ..., 2, 2, 2, ...]
            # where the number of repeats for each token position is equal to the number of features for that token
            all_token_positions.append(
                np.repeat(
                    np.arange(
                        global_position,
                        global_position + sequence_length,
                        dtype=np.uint64,
                    ),
                    feature_indices,
                )
            )

            # collapse feature_ids and activations to 1D arrays
            all_feature_ids.append(
                act.feature_ids.reshape(-1).astype(np.uint32, copy=False)
            )

            # collapse activations to 1D array
            all_values.append(
                act.activations.reshape(-1).astype(np.float16, copy=False)
            )

            # update global position for next prompt
            global_position += sequence_length

        token_positions_flat = np.concatenate(all_token_positions)
        feature_ids_flat = np.concatenate(all_feature_ids)
        values_flat = np.concatenate(all_values)

        return token_positions_flat, feature_ids_flat, values_flat

    def __init_feature_index_phase2(
        self,
        token_positions_flat: NDArray[np.uint64],
        feature_ids_flat: NDArray[np.uint32],
        values_flat: NDArray[np.float16],
        hidden_size: int,
    ):
        # sort by feature_id
        # order = np.argsort(feature_ids_flat, kind="stable")
        order = np.lexsort((-values_flat.astype(np.float32), feature_ids_flat))
        token_positions_sorted = token_positions_flat[order]
        values_sorted = values_flat[order]

        counts = np.bincount(feature_ids_flat, minlength=hidden_size)
        feature_offsets = np.zeros(hidden_size + 1, dtype=np.uint64)
        feature_offsets[1:] = np.cumsum(counts, dtype=np.uint64)

        return token_positions_sorted, feature_offsets, values_sorted

    def save_index(self, save_dir: str | Path):
        np.savez_compressed(
            save_dir,
            prompt_ids=self.prompt_ids,
            token_ids=self.token_ids,
            prompt_token_offsets=self.prompt_token_offsets,
            token_positions=self.token_positions,
            feature_offsets=self.feature_offsets,
            activation_values=self.activation_values,
        )

    @classmethod
    def load_index(cls, index_path: str | Path, mmap: bool = True):
        data = np.load(index_path, mmap_mode="r" if mmap else None)
        instance = cls.__new__(cls)
        instance.prompt_ids = data["prompt_ids"]
        instance.token_ids = data["token_ids"]
        instance.prompt_token_offsets = data["prompt_token_offsets"]
        instance.token_positions = data["token_positions"]
        instance.feature_offsets = data["feature_offsets"]
        instance.activation_values = data["activation_values"]
        return instance

    def _get_prompt(
        self,
        center_token_global_position: int,
    ) -> Prompt:
        prompt_index = (
            np.searchsorted(
                self.prompt_token_offsets, center_token_global_position, side="right"
            )
            - 1
        )
        prompt_start = int(self.prompt_token_offsets[prompt_index])
        prompt_end = int(self.prompt_token_offsets[prompt_index + 1])

        return Prompt(
            prompt_id=int(self.prompt_ids[prompt_index]),
            query_token_position=int(center_token_global_position - prompt_start),
            token_ids=self.token_ids[prompt_start:prompt_end],
        )

    def _top_tokens(self, feature_id: int, k: int, offset: int = 0):
        start = int(self.feature_offsets[feature_id])
        end = int(self.feature_offsets[feature_id + 1])

        if offset > 0:
            start += offset
            if start >= end:
                return TopKActivations(
                    feature_id=feature_id,
                    global_token_positions=np.array([], dtype=np.uint64),
                    activation_values=np.array([], dtype=np.float16),
                )
        top_end = min(end, start + k)
        return TopKActivations(
            feature_id=feature_id,
            global_token_positions=self.token_positions[start:top_end],
            activation_values=self.activation_values[start:top_end],
        )

    def _activations_for_span(
        self,
        feature_id: int,
        token_position_global_spans: list[tuple[int, int]],
    ) -> list[NDArray[np.float32]]:
        feature_start = self.feature_offsets[feature_id]
        feature_end = self.feature_offsets[feature_id + 1]
        token_positions = self.token_positions[feature_start:feature_end]
        activations = self.activation_values[feature_start:feature_end]

        position_to_activation = dict(
            zip(token_positions.astype(int).tolist(), activations.tolist())
        )
        results = []
        for start, end in token_position_global_spans:
            results.append(
                np.asarray(
                    [position_to_activation.get(pos, 0.0) for pos in range(start, end)],
                    dtype=np.float32,
                )
            )
        return results

    def get_topk_activating_examples(
        self,
        feature_id: int,
        k: int,
        window_size: int = 5,
        offset: int = 0,
    ):
        topk_activations = self._top_tokens(feature_id=feature_id, k=k, offset=offset)

        prompt_ids_list = []
        token_ids_list = []
        target_token_positions_list = []
        spans = []

        for _global_position in topk_activations.global_token_positions:
            global_position = int(_global_position)
            prompt = self._get_prompt(center_token_global_position=global_position)
            prompt_ids_list.append(prompt.prompt_id)

            global_prompt_start: int = global_position - int(
                prompt.query_token_position
            )
            global_prompt_end: int = int(global_prompt_start) + len(prompt.token_ids)

            global_span_start: int = max(
                global_prompt_start, global_position - window_size
            )
            global_span_end: int = min(
                global_prompt_end, global_position + window_size + 1
            )

            local_span_start: int = global_span_start - global_prompt_start
            local_span_end: int = global_span_end - global_prompt_start

            token_ids_list.append(prompt.token_ids[local_span_start:local_span_end])
            target_token_positions_list.append(
                prompt.query_token_position - local_span_start
            )
            spans.append((global_span_start, global_span_end))

        activations = self._activations_for_span(
            feature_id=feature_id,
            token_position_global_spans=spans,
        )

        return [
            ActivationExample(
                token_ids=token_ids,
                target_token_position=target_token_position,
                activation_values=activations[i],
                prompt_id=prompt_id,
            )
            for i, (token_ids, target_token_position, prompt_id) in enumerate(
                zip(token_ids_list, target_token_positions_list, prompt_ids_list)
            )
        ]

    @property
    def hidden_size(self) -> int:
        return len(self.feature_offsets) - 1

    def get_features_info_all(self) -> list[dict]:
        return [
            self.get_feature_info(feature_id) for feature_id in range(self.hidden_size)
        ]

    def get_feature_info(self, feature_id: int) -> dict:
        start = int(self.feature_offsets[feature_id])
        end = int(self.feature_offsets[feature_id + 1])
        values = self.activation_values[start:end].astype(np.float32, copy=False)
        if len(values) == 0:
            max_activation = 0.0
            mean_activation = 0.0
        else:
            max_activation = float(values.max())
            mean_activation = float(values.mean())
        return {
            "feature_id": feature_id,
            "max_activation": max_activation,
            "mean_activation": mean_activation,
            "activation_frequency": float(len(values)) / max(1, len(self.token_ids)),
            "num_fires": len(values),
        }

    def get_histogram(self, feature_id: int) -> list[dict]:
        if feature_id < 0 or feature_id >= self.hidden_size:
            return []
        start = int(self.feature_offsets[feature_id])
        end = int(self.feature_offsets[feature_id + 1])
        values = self.activation_values[start:end].astype(np.float32, copy=False)
        if len(values) == 0:
            return [{"bin_start": 0.0, "bin_end": 1.0, "count": 0}]
        counts, edges = np.histogram(
            values, bins=20, range=(0.0, max(float(values.max()), 1e-6))
        )
        return [
            {
                "bin_start": float(edges[i]),
                "bin_end": float(edges[i + 1]),
                "count": int(counts[i]),
            }
            for i in range(len(counts))
        ]
