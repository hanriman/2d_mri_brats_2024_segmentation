import sys
from pathlib import Path
from unittest.mock import patch

# Ensure repository root is on sys.path so scripts package is importable
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from scripts.evaluate import parse_args


def test_evaluate_parse_args_defaults():
    test_args = ["evaluate.py"]
    with patch.object(sys, "argv", test_args):
        args = parse_args()
        assert args.decoder_type is None
        assert args.encoder_source == "target"
        assert args.batch_size == 8
        assert args.device == "auto"
        assert args.deterministic is False


def test_evaluate_parse_args_custom():
    test_args = [
        "evaluate.py",
        "--decoder_type", "multiscale",
        "--encoder_source", "context",
        "--batch_size", "16",
        "--max_batches", "5",
        "--deterministic",
    ]
    with patch.object(sys, "argv", test_args):
        args = parse_args()
        assert args.decoder_type == "multiscale"
        assert args.encoder_source == "context"
        assert args.batch_size == 16
        assert args.max_batches == 5
        assert args.deterministic is True


def test_evaluate_low_data_parse_args():
    from scripts.evaluate_low_data import parse_args as parse_low_data_args
    test_args = [
        "evaluate_low_data.py",
        "--seeds", "42", "43", "44",
        "--epochs", "5",
        "--batch_size", "4",
    ]
    with patch.object(sys, "argv", test_args):
        args = parse_low_data_args()
        assert args.seeds == [42, 43, 44]
        assert args.epochs == 5
        assert args.batch_size == 4

