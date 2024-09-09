import yaml


def get_config(config_path, model='transformer'):
    """ Get configuration from yaml file, complementing default configuration

    Args:
        config_path (str): path to configuration file
        model (str): model name
            default: 'transformer'
            choices: ['transformer']

    Returns:
        config (dict): configuration
    """
    # default configuration
    if model == 'transformer':
        default_config = {
            # model configuration
            'model': {
                'src_tokenizer': 'bert-base-uncased',
                'tgt_tokenizer': 'cl-tohoku/bert-base-japanese',
                'N': 6,
                'num_heads': 8,
                'd_model': 512,
                'd_ff': 2048,
                'dropout': 0.1,
                'device': 'gpu',
                'max_len': 512,
                'eps': 1e-6,
            },
            # training configuration
            'training': {
                'batch_size': 32,
                'epochs': 10,
                'lr': 1e-4,
                'weight_decay': 5e-4,
                'warmup_steps': 100,
                'clip_grad_norm': 1.0,
                'save_dir': 'results',
                'resume': None,
                'tensorboard': False,
                'log_dir': 'logs',
            },
            # data configuration
            'data': {
                'train_data': 'data/train.txt',
                'val_data': 'data/dev.txt',
                'test_data': 'data/test.txt',
            }
        }
    else:
        raise ValueError(f'Invalid model name: {model}')

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # complement default configuration
    for key in default_config:
        if key not in config:
            config[key] = default_config[key]
        else:
            if not isinstance(config[key], dict):
                continue
            for sub_key in default_config[key]:
                if sub_key not in config[key]:
                    config[key][sub_key] = default_config[key][sub_key]

    return config
