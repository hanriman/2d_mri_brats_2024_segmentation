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
        "--seed", "42",
        "--epochs", "5",
        "--batch_size", "4",
    ]
    with patch.object(sys, "argv", test_args):
        args = parse_low_data_args()
        assert args.seed == 42
        assert args.epochs == 5
        assert args.batch_size == 4


def test_train_jepa_parse_args():
    from scripts.train_jepa import parse_args as parse_train_args
    test_args = [
        "train_jepa.py",
        "--model_type", "visreg_jepa",
        "--center_weight", "0.5",
        "--scale_weight", "1.5",
        "--shape_weight", "2.0",
        "--scale_loss_type", "hinge",
    ]
    with patch.object(sys, "argv", test_args):
        args = parse_train_args()
        assert args.model_type == "visreg_jepa"
        assert args.center_weight == 0.5
        assert args.scale_weight == 1.5
        assert args.shape_weight == 2.0
        assert args.scale_loss_type == "hinge"
        assert args.normalize_measure is True

    # Test sigreg options and no_normalize_measure flag
    test_sigreg_args = [
        "train_jepa.py",
        "--model_type", "sigreg_jepa",
        "--sigreg_weight", "0.75",
        "--no_normalize_measure",
    ]
    with patch.object(sys, "argv", test_sigreg_args):
        args_sig = parse_train_args()
        assert args_sig.model_type == "sigreg_jepa"
        assert args_sig.sigreg_weight == 0.75
        assert args_sig.normalize_measure is False


def test_train_downstream_parse_args():
    from scripts.train_downstream import parse_args as parse_downstream_args
    test_args = [
        "train_downstream.py",
        "--model_type", "visreg_jepa",
        "--pretrained_ckpt", "outputs/checkpoints/custom_model.pt",
        "--decoder_type", "multiscale",
        "--encoder_source", "context",
        "--p_drop", "0.25",
        "--amp",
    ]
    with patch.object(sys, "argv", test_args):
        args = parse_downstream_args()
        assert args.model_type == "visreg_jepa"
        assert args.pretrained_ckpt == "outputs/checkpoints/custom_model.pt"
        assert args.decoder_type == "multiscale"
        assert args.encoder_source == "context"
        assert args.p_drop == 0.25
        assert args.amp is True


def test_evaluate_ood_parse_args():
    from scripts.evaluate_ood import parse_args as parse_ood_args
    test_args = [
        "evaluate_ood.py",
        "--decoder_type", "multiscale",
        "--exp_version", "custom_ood",
        "--amp",
        "--batch_size", "32",
    ]
    with patch.object(sys, "argv", test_args):
        args = parse_ood_args()
        assert args.decoder_type == "multiscale"
        assert args.exp_version == "custom_ood"
        assert args.amp is True
        assert args.batch_size == 32


def test_evaluate_men_rt_ood_parse_args():
    from scripts.evaluate_men_rt_ood import parse_args as parse_men_args
    test_args = [
        "evaluate_men_rt_ood.py",
        "--decoder_type", "multiscale",
        "--exp_version", "custom_men_ood",
        "--amp",
        "--batch_size", "32",
    ]
    with patch.object(sys, "argv", test_args):
        args = parse_men_args()
        assert args.decoder_type == "multiscale"
        assert args.exp_version == "custom_men_ood"
        assert args.amp is True
        assert args.batch_size == 32



