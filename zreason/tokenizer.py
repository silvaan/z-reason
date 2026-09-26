"""Character-level tokenizer. Small and deterministic so experiments are reproducible."""

PAD, BOS, SEP, QRY = 0, 1, 2, 3
SPECIALS = ["<pad>", "<bos>", "<sep>", "<q>"]
CHARS = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789=+-*;:?<>(),. _")


class CharTokenizer:
    def __init__(self):
        self.itos = SPECIALS + CHARS
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        unk = self.stoi[" "]
        return [self.stoi.get(c, unk) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids if i >= len(SPECIALS))
