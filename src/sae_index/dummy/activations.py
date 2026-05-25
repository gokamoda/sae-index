import random
import string

import numpy as np

from sae_index.core.single import Activations

from .tokenizer import ASCIICharTokenizer


def generate_dummy_activations(
    num_prompts: int = 128,
    prompt_length: int = 512,
) -> list[Activations]:
    activations = []

    characters = string.ascii_lowercase + string.digits + " ."
    tokenizer = ASCIICharTokenizer()

    # random string of length `prompt_length`

    for i in range(num_prompts):
        prompt = "".join(random.choices(characters, k=prompt_length))
        _token_ids = tokenizer(prompt)

        _feature_ids = []
        _activations = []
        for j in range(len(_token_ids)):
            if j == 0:  # bos
                _feature_ids.append([2, _token_ids[j], _token_ids[j + 1]])
            elif j == len(_token_ids) - 1:  # eos
                _feature_ids.append([_token_ids[j - 1], _token_ids[j], 3])
            else:
                _feature_ids.append(
                    [_token_ids[j - 1], _token_ids[j], _token_ids[j + 1]]
                )
            _activations.append(np.random.rand(3).astype(np.float32))

        _token_ids = np.array(_token_ids, dtype=np.uint64)
        _feature_ids = np.array(_feature_ids, dtype=np.int64)
        _activations = np.array(_activations, dtype=np.float32)

        activations.append(
            Activations(
                prompt_id=i,
                token_ids=_token_ids,
                feature_ids=_feature_ids,
                activations=_activations,
            )
        )

    return activations
