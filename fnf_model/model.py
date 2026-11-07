import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig


class FNFTransformer(nn.Module):
    """Small causal Transformer language model; all weights start randomly initialized."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.block_size, config.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=4 * config.d_model,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=config.n_layers)
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = None):
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.block_size:
            raise ValueError(f"Sequence length {seq_len} exceeds block_size {self.config.block_size}")
        positions = torch.arange(seq_len, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
        # True entries are masked by PyTorch: tokens cannot attend to future positions.
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        x = self.blocks(x, mask=causal_mask)
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 100,
                 temperature: float = 0.8, top_k: int = 40, eos_id: int = 257):
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        self.eval()
        result = input_ids
        for _ in range(max_new_tokens):
            context = result[:, -self.config.block_size:]
            logits, _ = self(context)
            next_logits = logits[:, -1, :] / temperature
            if top_k > 0:
                k = min(top_k, next_logits.size(-1))
                threshold = torch.topk(next_logits, k).values[:, -1, None]
                next_logits = next_logits.masked_fill(next_logits < threshold, float("-inf"))
            probabilities = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            result = torch.cat((result, next_token), dim=1)
            if eos_id is not None and torch.all(next_token == eos_id):
                break
        return result
