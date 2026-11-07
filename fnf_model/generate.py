import argparse
from pathlib import Path

import torch

from .config import ModelConfig
from .model import FNFTransformer
from .tokenizer import ByteTokenizer


def main():
    parser = argparse.ArgumentParser(description="Generate FNF-focused text from a trained checkpoint.")
    parser.add_argument("--checkpoint", default="checkpoints/fnf_model_v2.pt")
    parser.add_argument("--prompt", default="FNF mod idea:")
    parser.add_argument("--tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}. Run training first.")
    device = torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = ModelConfig.from_dict(checkpoint["model_config"])
    model = FNFTransformer(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    tokenizer = ByteTokenizer()
    input_ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    output = model.generate(
        input_ids,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(tokenizer.decode(output[0].tolist()))


if __name__ == "__main__":
    main()
