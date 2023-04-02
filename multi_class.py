import argparse

from utils.preprocessor import PreprocessorClass
from utils.multi_class_trainer import MultiClassTrainer

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torchmetrics

def collect_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--accelerator", type=str, default='gpu')
    parser.add_argument("--gpu_id", type=int, default=1)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--max_length", type=int, default=100)
    parser.add_argument("--test_data_dir", type=str, default="bert_classification_sem3/data/testing.res")
    parser.add_argument("--train_data_dir", type=str, default="bert_classification_sem3/data/training.res")

    return parser.parse_args()

if __name__ == '__main__':
    args = collect_parser()

    prepo = PreprocessorClass(preprocessed_dir = "bert_classification_sem3/data",
                           train_data_dir= args.train_data_dir,
                           test_data_dir= args.test_data_dir,
                           batch_size = args.batch_size,
                           max_length = args.max_length)

    # Learning rate diganti 1e-3 ke 1e-5
    # model = MultiClassModel(
    #     n_out = 5,
    #     dropout = 0.3,
    #     lr = 1e-5,
    #     max_epoch = 10
    # )

    train_dataset, validation_dataset, test_dataset = prepo.preprocessor_manual()

    mclass_trainer = MultiClassTrainer( dropout = 0.1, 
                                        lr = 2e-5, 
                                        max_epoch = 10, 
                                        device = "cuda", 
                                        n_class= len(prepo.label2id))

    mclass_trainer.trainer(train_dataset, validation_dataset, test_dataset)

    # logger = TensorBoardLogger("logs", name="bert-multi-class")

    # trainer = pl.Trainer(
    #     accelerator= args.accelerator,
    #     devices = args.gpu_id,
    #     num_nodes=args.num_nodes,
    #     max_epochs = 10,
    #     default_root_dir = "bert_classification_sem3/checkpoints/class",
    #     logger= logger
    # )

    # trainer.fit(model, datamodule = dm)
    # # pred, true = trainer.predict(model = model, datamodule = dm)
    # hasil = trainer.predict(model= model, datamodule= dm)
    


