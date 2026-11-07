import unittest

import torch

from fnf_model.config import ModelConfig
from fnf_model.model import FNFTransformer
from fnf_model.tokenizer import ByteTokenizer, VOCAB_SIZE


class ModelTests(unittest.TestCase):
    def test_tokenizer_round_trip(self):
        tokenizer = ByteTokenizer()
        text = "FNF: Gracjan i Boyfriend"
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_forward_and_backward(self):
        config = ModelConfig(vocab_size=VOCAB_SIZE, block_size=16, d_model=32, n_heads=4, n_layers=1)
        model = FNFTransformer(config)
        x = torch.randint(0, VOCAB_SIZE, (2, 16))
        logits, loss = model(x, x)
        self.assertEqual(tuple(logits.shape), (2, 16, VOCAB_SIZE))
        self.assertIsNotNone(loss)
        loss.backward()
        self.assertIsNotNone(model.token_embedding.weight.grad)

    def test_rejects_long_context(self):
        config = ModelConfig(vocab_size=VOCAB_SIZE, block_size=8, d_model=32, n_heads=4, n_layers=1)
        model = FNFTransformer(config)
        with self.assertRaises(ValueError):
            model(torch.zeros((1, 9), dtype=torch.long))


if __name__ == "__main__":
    unittest.main()
