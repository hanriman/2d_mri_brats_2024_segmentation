import argparse

from brats_jepa.config import merge_config_with_args


def test_merge_config_overrides_defaults():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args([])

    config = {
        'epochs': 30,
        'batch_size': 16,
        'lr': 5e-5,
    }

    # Simulate running without explicit flags in sys.argv
    merged = merge_config_with_args(config, args, cli_args=[])
    assert merged.epochs == 30
    assert merged.batch_size == 16
    assert merged.lr == 5e-5


def test_cli_overrides_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args(['--epochs', '20'])

    config = {
        'epochs': 30,
        'batch_size': 16,
    }

    # Explicit --epochs was passed, batch_size was not
    merged = merge_config_with_args(config, args, cli_args=['--epochs', '20'])
    assert merged.epochs == 20
    assert merged.batch_size == 16


def test_name_maps_to_model_type():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='ijepa')
    args = parser.parse_args([])

    config = {
        'name': 'sigreg_jepa',
        'proj_dim': 128,
        'num_projections': 256,
    }

    merged = merge_config_with_args(config, args, cli_args=[])
    assert merged.model_type == 'sigreg_jepa'
    assert merged.name == 'sigreg_jepa'
    assert merged.proj_dim == 128
    assert merged.num_projections == 256


def test_nested_dict_merging():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='outputs')
    args = parser.parse_args([])

    config = {
        'paths': {
            'output_dir': 'custom_outputs',
            'checkpoints_dir': 'custom_outputs/checkpoints',
        }
    }

    merged = merge_config_with_args(config, args, cli_args=[])
    assert merged.output_dir == 'custom_outputs'
    assert merged.checkpoints_dir == 'custom_outputs/checkpoints'
    assert 'paths' in merged.paths or isinstance(merged.paths, dict)
