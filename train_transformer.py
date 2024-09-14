import argparse
import logging
import os

import torch
import torch.nn as nn
from transformer.models.transformer import Transformer
from transformer.dataset import TranslationDataset
from transformer.trainer import Trainer
from transformer.utils import get_config, get_logger
from transformers import AutoTokenizer

logger = get_logger(__name__, level=logging.INFO)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def main(config):
    src_tokenizer = AutoTokenizer.from_pretrained(config['model']['src_tokenizer'])
    tgt_tokenizer = AutoTokenizer.from_pretrained(config['model']['tgt_tokenizer'])

    # prepare dataset
    train_src, train_tgt = [], []
    with open(config['data']['train_data'], 'r') as f:
        for line in f:
            src, tgt = line.strip().split('\t')
            train_src.append(src)
            train_tgt.append(tgt)
    val_src, val_tgt = [], []
    with open(config['data']['val_data'], 'r') as f:
        for line in f:
            src, tgt = line.strip().split('\t')
            val_src.append(src)
            val_tgt.append(tgt)

    train_data = TranslationDataset(train_src, train_tgt)
    val_data = TranslationDataset(val_src, val_tgt)

    # prepare model
    model = Transformer(src_tokenizer, tgt_tokenizer, d_model=config['model']['d_model'], num_heads=config['model']['num_heads'],
                        d_ff=config['model']['d_ff'], N=config['model']['N'], dropout=config['model']['dropout'],
                        device=config['model']['device'], max_len=config['model']['max_len'], eps=config['model']['eps'])

    # prepare trainer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_tokenizer.pad_token_id)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['epochs'])
    trainer = Trainer(model, criterion, optimizer, scheduler, config)

    # train
    trainer.train(train_data, val_data, config['training']['resume'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='path to configuration file', required=True)
    args = parser.parse_args()

    config = get_config(args.config)
    main(config)
