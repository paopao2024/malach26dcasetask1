import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch
from torch.utils.data import DataLoader
import argparse
import torch.nn.functional as F
import transformers
from functools import partial
import secrets
import subprocess
import sys
import numpy as np


from datasets.audiodataset import get_val_set, get_training_set
from models.cnn import get_model
from models.mel import AugmentMelSTFT
from helpers.init import worker_init_fn
from helpers.mixup import mixup


class SimpleDCASELitModule(pl.LightningModule):
    """
    This is a Pytorch Lightening Module.
    It has several convenient abstractions, e.g., we don't have to specify all parts of the
    training loop (optimizer.step(), loss.backward()) ourselves.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config  # results from argparse and contains all configurations for our experiment
        # model to preprocess waveforms into log mel spectrograms
        self.mel = AugmentMelSTFT(n_mels=config.n_mels,
                                  sr=config.resample_rate,
                                  win_length=config.window_size,
                                  hopsize=config.hop_size,
                                  n_fft=config.n_fft,
                                  freqm=config.freqm,
                                  timem=config.timem,
                                  fmin=config.fmin,
                                  fmax=config.fmax,
                                  fmin_aug_range=config.fmin_aug_range,
                                  fmax_aug_range=config.fmax_aug_range
                                  )

        # our model to be trained on the log mel spectrograms
        self.model = get_model(in_channels=config.in_channels,
                               n_classes=config.n_classes,
                               base_channels=config.base_channels,
                               channels_multiplier=config.channels_multiplier
                               )

        # These are containers we can use to store metrics computed
        # in 'training_step' and 'validation_step'. The containers can then be processed
        # in 'on_train_epoch_end' and 'on_validation_epoch_end'.
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def mel_forward(self, x):
        """
        @param x: a batch of raw signals (waveform)
        return: a batch of log mel spectrograms
        """
        old_shape = x.size()
        x = x.reshape(-1, old_shape[2])  # for calculating mel spectrograms we remove the channel dimension
        x = self.mel(x)
        x = x.reshape(old_shape[0], old_shape[1], x.shape[1], x.shape[2])  # batch x channels x mels x time-frames
        return x

    def forward(self, x):
        """
        :param x: batch of raw audio signals (waveforms)
        :return: final model predictions
        """
        x = self.mel_forward(x)
        x = self.model(x)
        return x

    def configure_optimizers(self):
        """
        This is the way pytorch lightening requires optimizers and learning rate schedulers to be defined.
        The specified items are used automatically in the optimization loop (no need to call optimizer.step() yourself).
        :return: dict containing optimizer and learning rate scheduler
        """
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.config.lr,
                                     weight_decay=self.config.weight_decay)

        num_training_steps = self.trainer.estimated_stepping_batches
        # cosine learning rate schedule
        lr_scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler
        }

    def training_step(self, train_batch, batch_idx):
        """
        :param train_batch: contains one batch from train dataloader
        :param batch_idx: will likely not be used at all
        :return: a dict containing at least loss that is used to update model parameters, can also contain
                    other items that can be processed in 'training_epoch_end' to log other metrics than loss
        """
        x, y = train_batch  # we get a batch of raw audio signals and labels as defined by our dataset
        bs = x.size(0)
        x = self.mel_forward(x)  # we convert the raw audio signals into log mel spectrograms

        if args.mixup_alpha:
            # Apply Mixup, a very common data augmentation method
            rn_indices, lam = mixup(bs, args.mixup_alpha)  # get shuffled indices and mixing coefficients
            # send mixing coefficients to correct device and make them 4-dimensional
            lam = lam.to(x.device).reshape(bs, 1, 1, 1)
            # mix two spectrograms from the batch
            x = x * lam + x[rn_indices] * (1. - lam)
            # generate predictions for mixed log mel spectrograms
            y_hat = self.model(x)
            # mix the prediction targets using the same mixing coefficients
            samples_loss = (
                    F.cross_entropy(y_hat, y, reduction="none") * lam.reshape(bs) +
                    F.cross_entropy(y_hat, y[rn_indices], reduction="none") * (1. - lam.reshape(bs))
            )

        else:
            y_hat = self.model(x)
            # cross_entropy is used for multiclass problems
            # be careful when choosing the correct loss functions
            # read the documentation what input your loss function expects, e.g. for F.cross_entropy:
            # the logits (no softmax!) and the prediction targets (class indices)
            samples_loss = F.cross_entropy(y_hat, y, reduction="none")

        loss = samples_loss.mean()
        results = {"loss": loss}
        # log learning rate, number of epoch, loss on each step
        self.log('trainer/lr', self.trainer.optimizers[0].param_groups[0]['lr'])
        self.log('epoch', self.current_epoch)
        self.log("train/loss", loss.detach().cpu())
        return results['loss']

    def on_train_epoch_end(self):
        # do some stuff at the end of a train epoch
        # nothing to do in our case
        pass

    def validation_step(self, val_batch, batch_idx):
        # similar to 'training_step' but without any data augmentation
        # pytorch lightening takes care of 'with torch.no_grad()' and 'model.eval()'
        x, y = val_batch
        x = self.mel_forward(x)
        y_hat = self.model(x)
        samples_loss = F.cross_entropy(y_hat, y, reduction="none")

        # results dict contains only 'loss' for this example
        # you could, e.g., also store predictions and labels in the container
        results = {'loss': samples_loss}
        results = {k: v.cpu() for k, v in results.items()}
        self.validation_step_outputs.append(results)

    def on_validation_epoch_end(self):
        # process results from 'validation_step'
        # logging only 'loss' could be done much simpler than code piece below,
        # however this is a more general solution
        outputs = {k: [] for k in self.validation_step_outputs[0]}
        for step_output in self.validation_step_outputs:
            for k in step_output:
                outputs[k].append(step_output[k])
        for k in outputs:
            outputs[k] = torch.cat(outputs[k])

        avg_loss = outputs['loss'].mean()
        logs = {'val/loss': torch.as_tensor(avg_loss).cuda()}

        self.log_dict(logs, sync_dist=True)

        # clear container
        self.validation_step_outputs.clear()


def train(config):
    # logging is done using wandb
    wandb_logger = WandbLogger(
        project="MALACH_TEST_PIPELINE",
        notes="Test audio pipeline for MALACH.",
        tags=["MALACH25"],
        config=config,  # this logs all hyperparameters for us
        name=config.experiment_name
    )

    # train dataloader
    train_dl = DataLoader(dataset=get_training_set(config.cache_path, config.resample_rate, config.roll),
                          worker_init_fn=partial(worker_init_fn, seed=config.SEED),
                          generator=torch.Generator().manual_seed(config.SEED),
                          num_workers=config.num_workers,
                          batch_size=config.batch_size,
                          shuffle=True)

    # test loader
    val_dl = DataLoader(dataset=get_val_set(config.cache_path, config.resample_rate),
                        worker_init_fn=partial(worker_init_fn, seed=config.SEED),
                        num_workers=config.num_workers,
                        batch_size=config.batch_size)

    # create pytorch lightening module
    pl_module = SimpleDCASELitModule(config)
    # create the pytorch lightening trainer by specifying the number of epochs to train, the logger,
    # on which kind of device(s) to train and possible callbacks
    trainer = pl.Trainer(max_epochs=config.n_epochs,
                         logger=wandb_logger,
                         accelerator='auto')
    # start training and validation
    trainer.fit(pl_module, train_dataloaders=train_dl, val_dataloaders=val_dl)


if __name__ == '__main__':
    # simplest form of specifying hyperparameters using argparse
    # IMPORTANT: log hyperparameters to be able to reproduce you experiments!
    parser = argparse.ArgumentParser(description='Example of parser. ')

    # general
    parser.add_argument('--experiment_name', type=str, default="DCASE25")
    parser.add_argument('--num_workers', type=int, default=12)  # number of workers for dataloaders
    parser.add_argument('--SEED', type=int, default=None)
    parser.add_argument('--make_deterministic', default=False, action='store_true')

    # dataset
    # location to store resample waveform
    parser.add_argument('--cache_path', type=str, default="datasets/example_data/cached")
    parser.add_argument('--roll', default=False, action='store_true')  # rolling waveform over time

    # model
    parser.add_argument('--n_classes', type=int, default=10)  # classification model with 'n_classes' output neurons
    # spectrograms have 1 input channel (RGB images would have 3)
    parser.add_argument('--in_channels', type=int, default=1)
    # adapt the complexity of the neural network
    parser.add_argument('--base_channels', type=int, default=16)
    parser.add_argument('--channels_multiplier', type=int, default=2)

    # training
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--mixup_alpha', type=float, default=0.0)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    # learning rate + schedule
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--num_warmup_steps', type=int, default=100)

    # preprocessing
    parser.add_argument('--resample_rate', type=int, default=32000)
    parser.add_argument('--window_size', type=int, default=800)  # in samples (corresponds to 25 ms)
    parser.add_argument('--hop_size', type=int, default=320)  # in samples (corresponds to 10 ms)
    parser.add_argument('--n_fft', type=int, default=1024)  # length (points) of fft, e.g. 1024 point FFT
    parser.add_argument('--n_mels', type=int, default=128)  # number of mel bins
    parser.add_argument('--freqm', type=int, default=0)  # mask up to 'freqm' spectrogram bins
    parser.add_argument('--timem', type=int, default=0)  # mask up to 'timem' spectrogram bins
    parser.add_argument('--fmin', type=int, default=0)  # mel bins are created for freqs. between 'fmin' and 'fmax'
    parser.add_argument('--fmax', type=int, default=None)
    parser.add_argument('--fmin_aug_range', type=int, default=1)  # data augmentation: vary 'fmin' and 'fmax'
    parser.add_argument('--fmax_aug_range', type=int, default=1000)

    args = parser.parse_args()

    if args.SEED is None:
        args.SEED = secrets.randbelow(2**32)

    pl.seed_everything(args.SEED, workers=True)

    if args.make_deterministic:
        # you may need to set the environment variable 'CUBLAS_WORKSPACE_CONFIG=:4096:8' for this to work
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # add commit hash to logger
    try:
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except Exception:
        commit_hash = "unknown"

    args.commit_hash = commit_hash  # include in logged config

    # save library versions for reproducibility
    args.versions = {
        "python": sys.version.split(" ")[0],
        "torch": torch.__version__,
        "pytorch_lightning": pl.__version__,
        "numpy": np.__version__
    }

    train(args)
