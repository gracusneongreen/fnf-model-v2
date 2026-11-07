import argparse
import random
from pathlib import Path

import torch

from .config import ModelConfig
from .data import TokenBlockDataset, load_texts
from .model import FNFTransformer
from .tokenizer import VOCAB_SIZE


def parse_args():
    parser = argparse.ArgumentParser(description="Train FNF Model V2 from random initialization.")
    parser.add_argument("--data", default="data/fnf_train.jsonl")
    parser.add_argument("--out", default="checkpoints/fnf_model_v2.pt")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--device", choices=["auto", "cpu"], default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.steps < 1 or args.batch_size < 1:
        raise SystemExit("--steps and --batch-size must be positive integers")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cpu" if args.device == "cpu" or not torch.cuda.is_available() else "cuda")
    print(f"Device: {device} (Intel Iris Xe uses CPU in this first prototype)")

    texts = load_texts(args.data)
    dataset = TokenBlockDataset(texts, args.block_size)
    config = ModelConfig(
        vocab_size=VOCAB_SIZE,
        block_size=args.block_size,
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
    )
    model = FNFTransformer(config).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Training from random initialization: {parameter_count:,} parameters")
    print(f"Training tokens: {dataset.tokens.numel():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    model.train()
    for step in range(1, args.steps + 1):
        indices = torch.randint(0, len(dataset), (args.batch_size,))
        x = torch.stack([dataset[int(i)][0] for i in indices]).to(device)
        y = torch.stack([dataset[int(i)][1] for i in indices]).to(device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.eval_interval == 0 or step == args.steps:
            print(f"step {step:5d}/{args.steps} | train loss {loss.item():.4f}")

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": config.to_dict(),
            "training": {
                "steps": args.steps,
                "seed": args.seed,
                "data": str(args.data),
                "initialized_from_pretrained": False,
            },
        },
        output,
    )
    print(f"Saved checkpoint: {output.resolve()}")


if __name__ == "__main__":
    main()
