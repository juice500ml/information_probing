from argparse import ArgumentParser
from datetime import datetime
from os import getcwd
from pathlib import Path

from pytorch_lightning import Trainer, seed_everything, loggers

from model import Wav2Vec2WithProbe
from dataset import CommonPhoneDataModule


def main(args):
    datamodule = CommonPhoneDataModule(**args)
    model = Wav2Vec2WithProbe(num_classes=datamodule.num_classes, **args)

    seed_everything(seed=42, workers=True)
    trainer = Trainer(
        accelerator=args["accelerator"],
        devices=args["devices"],
        fast_dev_run=args["fast_dev_run"],
        max_epochs=args["max_epochs"],
        deterministic=True,
        check_val_every_n_epoch=1,
        default_root_dir=f"{getcwd()}/exps/{args['dataset_path']}/layer_index={args['layer_index']}/{datetime.today().isoformat()}"
    )
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    parser = ArgumentParser()

    # DataModule
    parser.add_argument("--dataset_path", type=Path, help="Path to dataset csv file")
    parser.add_argument("--model_name", default="facebook/wav2vec2-base", help="Huggingface transformers model name")
    parser.add_argument("--max_length", type=float, default=10.0, help="Max audio length in seconds")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of multiprocessing workers for dataloading")

    # Model
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for AdamW")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay for AdamW")
    parser.add_argument("--tuning_type", choices=["linear", "finetune"], help="Type of training (linear: linear probing, finetune: fine-tuning)")
    parser.add_argument("--layer_index", type=int, help="Transformer layer index to use")

    # Trainer
    parser.add_argument("--accelerator",  default="gpu", help="gpu or cpu")
    parser.add_argument("--devices", default=1, help="# of gpus")
    parser.add_argument("--fast_dev_run", help="True if debug mode", default=False)
    parser.add_argument("--max_epochs", type=int, default=5, help="Maximum epochs to train")

    args = parser.parse_args()
    main(vars(args))
