import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .tokenizer import ByteTokenizer


def load_texts(path: str | Path) -> list[str]:
    """Read JSONL rows with a 'text' field, or plain UTF-8 text files."""
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        texts = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
                if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                    raise ValueError(f"Line {line_number} must be an object with a string 'text' field")
                texts.append(row["text"])
        return texts
    return [path.read_text(encoding="utf-8")]


class TokenBlockDataset(Dataset):
    def __init__(self, texts: list[str], block_size: int):
        tokenizer = ByteTokenizer()
        token_ids = []
        for text in texts:
            token_ids.extend(tokenizer.encode(text, add_bos=True, add_eos=True))
        if len(token_ids) < block_size + 2:
            raise ValueError(
                f"Dataset has {len(token_ids)} tokens; need at least {block_size + 2}. "
                "Add more FNF training examples or reduce --block-size."
            )
        self.tokens = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return max(1, len(self.tokens) - self.block_size)

    def __getitem__(self, index):
        chunk = self.tokens[index:index + self.block_size + 1]
        return chunk[:-1], chunk[1:]
