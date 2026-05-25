class ASCIICharTokenizer:
    def __init__(self):
        pass

    def __call__(self, text: str) -> list[int]:
        return [ord(char) for char in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(token_id) for token_id in token_ids)

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        return [ord(token) for token in tokens]

    def convert_ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        return [chr(token_id) for token_id in token_ids]
