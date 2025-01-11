import argparse
import os
import json

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from task import DCI_task


def parse_args():
    parser = argparse.ArgumentParser()
    # data args
    parser.add_argument("--ann_dir", type=str, default='./data_files')
    parser.add_argument("--pseudolabel_dir", type=str, default="/data/wzc/data/DCI-OSC/videoclip_pseudolabel")
    parser.add_argument("--feat_dir", type=str, default='/data/wzc/data/DCI-OSC')
    parser.add_argument("--sc_list", type=str, nargs='+', default=[''])
    # model args
    parser.add_argument("--transformer_heads", type=int, default=4)
    parser.add_argument("--transformer_layers", type=int, default=3)
    parser.add_argument("--transformer_dim", type=int, default=512)
    parser.add_argument("--transformer_dropout", type=float, default=0.1)
    # training args
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--wd", type=float, default=0.00001)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--n_epochs", type=int, default=30)
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--log_name", type=str, default="")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--a", type=int, default=1)
    parser.add_argument("--b", type=int, default=1)
    parser.add_argument("--tau", type=int, default=1)

    parser.add_argument("--det", default=1, type=int, choices=[0, 1])

    return parser.parse_args()


def main():
    log_dir = os.path.join(args.log_dir, args.log_name)
    os.makedirs(log_dir, exist_ok=True)
    print('Logging to:', log_dir)
    with open(f'{log_dir}/args.json', 'w') as f:
        json.dump(vars(args), f, indent=4)

    task = DCI_task(args)

    checkpoint_callbacks = []
    if hasattr(task, 'metric_name_list'):
        for name in task.metric_name_list:
            checkpoint_callbacks.append(
                ModelCheckpoint(
                    monitor=name,
                    filename='model-{epoch:02d}-{' + name + ':.4f}' + f' {args.lr}',
                    save_top_k=1,
                    mode='max',
                )
            )

    trainer = pl.Trainer(
        devices=args.gpus,
        accelerator="gpu",
        max_epochs=args.n_epochs,
        default_root_dir=log_dir,
        logger=TensorBoardLogger(save_dir=log_dir),
        callbacks=checkpoint_callbacks,
        num_sanity_val_steps=0,
    )

    if args.ckpt != "":
        task = DCI_task.load_from_checkpoint(checkpoint_path=args.ckpt, args=args)
        print(f'Evaluating {args.ckpt}')
        trainer.validate(task)
    else:
        trainer.fit(task)


if __name__ == '__main__':
    args = parse_args()

    main()
