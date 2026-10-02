"""Command-line interface for SPACE."""
import argparse
import logging

from .checkpoint import space_calibrate


def main(argv=None):
    parser = argparse.ArgumentParser(description="Calibrate an SFT checkpoint against its pre-SFT geometry.")
    parser.add_argument("--pre", required=True, help="Local directory or Hugging Face model ID")
    parser.add_argument("--post", required=True, help="Full SFT safetensors checkpoint")
    parser.add_argument("--output", required=True, help="New output directory")
    parser.add_argument("--rho", type=float, default=0.5, help="Cumulative squared singular-value energy (default: 0.5)")
    parser.add_argument("--alpha", type=float, default=1.0, help="Calibration strength (default: 1)")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = space_calibrate(args.pre, args.post, args.output, rho=args.rho, alpha=args.alpha, device=args.device)
    print(f"Saved SPACE checkpoint to {result}")


if __name__ == "__main__":
    main()
