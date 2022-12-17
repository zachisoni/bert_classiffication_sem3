import argparse

from utils.preprocessor import PreprocessorClass
from models.multi_class_model import MultiClassModel

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

def collect_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--accelerator", type=str, default='gpu')
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--max_length", type=int, default=100)
    parser.add_argument("--test_data_dir", type=str, default="data/testing.res")
    parser.add_argument("--test_data_dir", type=str, default="data/training.res")

    return parser.parse_args()

if __name__ == '__main__':
    dm = PreprocessorClass(preprocessed_dir = "bert_classification_sem3/sata/preprocessed",
                           batch_size = 100,
                           max_length = 100)

    # Learning rate diganti 1e-3 ke 1e-5
    model = MultiClassModel(
        n_out = 5,
        dropout = 0.3,
        lr = 1e-5
    )

    logger = TensorBoardLogger("logs", name="bert-multi-class")

    trainer = pl.Trainer(
        gpus = 1,
        max_epochs = 10,
        default_root_dir = "bert_classification_sem3/checkpoints/class"
    )

    trainer.fit(model, datamodule = dm)
    # pred, true = trainer.predict(model = model, datamodule = dm)

