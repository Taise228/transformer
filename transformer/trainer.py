import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from transformer.utils import get_logger

logger = get_logger(__name__)


class Trainer:
    """ Trainer class
    """

    def __init__(self, model, criterion, optimizer, scheduler, config):
        """ Instantiating Trainer class
        Args:
            model (nn.Module): model to train
            criterion (torch.nn.modules.loss._Loss): loss function
            optimizer (torch.optim): optimizer
            scheduler (torch.optim.lr_scheduler): learning rate scheduler
            config (dict): configuration
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config

        self.device = self.config['model']['device']
        self.model.to(self.device)

        self.save_dir = self.config['training']['save_dir']
        self.warmup_epochs = self.config['training']['warmup_epochs']

        if self.config['training']['tensorboard']:
            self.writer = SummaryWriter(self.config['training']['log_dir'])
        else:
            self.writer = None

    def train(self, train_data, valid_data, resume=None):
        """ Train model
        Args:
            train_data (torch.utils.data.Dataset): training dataset
            valid_data (torch.utils.data.Dataset): validation dataset
            resume (str): path to resume training
        """
        train_loader = DataLoader(train_data, batch_size=self.config['training']['batch_size'], shuffle=True,
                                  drop_last=True, num_workers=4, pin_memory=True)
        valid_loader = DataLoader(valid_data, batch_size=self.config['training']['batch_size'], shuffle=False,
                                  drop_last=False, num_workers=4, pin_memory=True)

        if resume:
            self.resume(resume)

        best_loss = float('inf')
        for epoch in range(self.config['training']['epochs']):
            self.writer.add_scalar('Learning Rate', self.optimizer.param_groups[0]['lr'], epoch + 1)

            self.model.train()
            total_loss = 0
            for src, tgt in train_loader:
                src = self.model.src_tokenizer(src, padding=True, truncation=True, return_tensors='pt')['input_ids']
                tgt = self.model.tgt_tokenizer(tgt, padding=True, truncation=True, return_tensors='pt')['input_ids']
                src, tgt = src.to(self.device), tgt.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(src, tgt[:, :-1])['logits']   # shape: (batch_size, tgt_len, vocab_size)
                output = output.reshape(-1, output.size(2))   # shape: (batch_size * tgt_len, vocab_size)
                tgt = tgt[:, 1:].reshape(-1)   # shape: (batch_size * tgt_len), dtype: torch.int64.
                # CrossEntropyLoss converts the target to one-hot encoding internally.
                loss = self.criterion(output, tgt)
                loss.backward()

                # gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['clip_grad_norm'])

                self.optimizer.step()
                total_loss += loss.item()
            logger.info(f'Epoch: {epoch+1}, Loss: {total_loss:.4f}')
            if self.writer:
                # tensorboard
                self.writer.add_scalar('Loss/train', total_loss, epoch + 1)

            self.model.eval()
            total_loss = 0
            with torch.no_grad():
                for src, tgt in valid_loader:
                    src, tgt = src.to(self.device), tgt.to(self.device)
                    output = self.model(src, tgt[:, :-1])
                    output = output['logits'].reshape(-1, output['logits'].size(2))
                    tgt = tgt[:, 1:].reshape(-1)
                    loss = self.criterion(output, tgt)
                    total_loss += loss.item()
            logger.info(f'Epoch: {epoch+1}, Validation Loss: {total_loss:.4f}')
            if self.writer:
                # tensorboard
                self.writer.add_scalar('Loss/valid', total_loss, epoch + 1)
            if total_loss < best_loss:
                best_loss = total_loss
                self.save('best.pth', loss=best_loss)

            if epoch > self.warmup_epochs:
                self.scheduler.step()

        self.save('last.pth', loss=total_loss)

    def resume(self, path):
        """ Resume training
        Args:
            path (str): path to model
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])


    def save(self, path, loss=None):
        """ Save model
        Args:
            path (str): path to save model
            loss (float): loss value to save
        """
        save_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
            'loss': loss
        }
        torch.save(save_dict, path)
        logger.info(f'Model is saved at {path}')

    def close(self):
        """ Close tensorboard writer
        """
        if self.writer:
            self.writer.close()
        logger.info('Tensorboard writer is closed')
