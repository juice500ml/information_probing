from argparse import ArgumentParser
from datetime import datetime
from os import getcwd
from pathlib import Path

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from model import Wav2Vec2WithProbe
from dataset import CommonPhoneDataModule


def main(args):
    datamodule = CommonPhoneDataModule(**args)
    model = Wav2Vec2WithProbe(num_classes=datamodule.num_classes, **args)

    seed_everything(seed=42, workers=True)
    checkpoint_callback = ModelCheckpoint(monitor="val/loss")

    trainer = Trainer(
        accelerator=args["accelerator"],
        devices=args["devices"],
        fast_dev_run=args["fast_dev_run"],
        max_epochs=args["max_epochs"],
        val_check_interval=args["val_check_interval"],
        callbacks=[checkpoint_callback],
        deterministic=True,
        default_root_dir=f"{getcwd()}/exps/{args['exp_name']}_{datetime.today().isoformat()}"
    )
    trainer.fit(model, datamodule=datamodule)
    trainer.test(ckpt_path="best", datamodule=datamodule)


if __name__ == "__main__":
    parser = ArgumentParser()

    # DataModule
    parser.add_argument("--dataset_path", type=Path, help="Path to dataset csv file")
    parser.add_argument("--model_name", default="facebook/wav2vec2-base", help="Huggingface transformers model name")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=5, help="Number of multiprocessing workers for dataloading")

    # Model
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for AdamW")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay for AdamW")
    parser.add_argument("--tuning_type", choices=["linear", "finetune"], help="Type of training (linear: linear probing, finetune: fine-tuning)")
    parser.add_argument("--layer_index", type=int, help="Transformer layer index to use")
    parser.add_argument("--probe_type", default="linear", choices=["linear", "mlp1", "mlp2"], help="Structure for the probe (Hewitt & Liang 2019)")
    parser.add_argument("--probe_hidden_dim", default=0, type=int, help="Hidden dimension for the MLP probes (Hewitt & Liang 2019)")

    # Trainer
    parser.add_argument("--exp_name", type=str, default="test", help="Experiment name (Folder to store the results)")
    parser.add_argument("--accelerator",  default="gpu", help="gpu or cpu")
    parser.add_argument("--devices", default=1, help="# of gpus")
    parser.add_argument("--fast_dev_run", help="True if debug mode", default=False)
    parser.add_argument("--max_epochs", type=int, default=10, help="Maximum epochs to train")
    parser.add_argument("--val_check_interval", type=float, default=1.0, help="Validation interval (1.0 = Once per epoch, 0.25 = 4 times per epoch)")

    args = parser.parse_args()
    main(vars(args))
