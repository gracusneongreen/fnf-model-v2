"""A dependency-free UTF-8 byte tokenizer.

IDs 0..255 represent bytes. 256=BOS, 257=EOS, 258=PAD.
This keeps the first prototype simple; a learned subword tokenizer can be added later.
"""
from typing import Iterable

BOS_ID = 256
EOS_ID = 257
PAD_ID = 258
VOCAB_SIZE = 259


class ByteTokenizer:
    vocab_size = VOCAB_SIZE

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> list[int]:
        ids = list(text.encode("utf-8"))
        if add_bos:
            ids.insert(0, BOS_ID)
        if add_eos:
            ids.append(EOS_ID)
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        raw = bytearray()
        for token_id in ids:
            token_id = int(token_id)
            if token_id in (BOS_ID, EOS_ID, PAD_ID) and skip_special:
                continue
            if 0 <= token_id <= 255:
                raw.append(token_id)
        return raw.decode("utf-8", errors="replace")
