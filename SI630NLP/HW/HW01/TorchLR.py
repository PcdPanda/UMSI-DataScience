import pickle
from typing import List, Callable, Dict
from Utils import *
import altair as alt
import math
import numpy as np
import pandas as pd
import time
import logging
import torch
from torch import nn
import tqdm
from scipy import sparse


logger = logging.getLogger("PyTorch Logistic Regression Logger")
logging.basicConfig(level=logging.DEBUG)


class LogisticRegression(nn.Module):
    def __init__(self, processor: int):
        """A two level neruo network for logistic regression classifier
        
        Args:
            processor(Preprocessor): A processor with vocabular set 

        Attributes:
            processor(Preprocessor):
            layer_stack (nn.Sequential): The two level neuro network structure
        """
        super().__init__()
        self.processor = processor
        self.layer_stack = nn.Sequential(
            nn.Linear(len(self.processor.voc) + 1, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        """The forward method for training"""
        return self.layer_stack(x).flatten()

    def get_token_weights(self):
        """Sort the words by their weights
        
        Returns:
            (List[float, str]): Sorted tokens list by their weight
        """
        pri = dict()
        for k, v in self.processor.voc.items():
            pri[v] = k
        return sorted(
            [
                (val, pri[i])
                for i, val in enumerate(
                    next(self.parameters())[0].detach().numpy().flatten()[:-1]
                )
            ],
            key=lambda x: -abs(x[0]),
        )


def train(
    train_X: np.ndarray,
    train_Y: np.ndarray,
    model: LogisticRegression,
    lr: float = 5e-5,
    epoch_num: int = 5,
    epoch_step: int = 1000,
    optimizer: torch.optim.Optimizer = None,
    fig_path: str = "",
) -> Dict[str, object]:
    """Train the model through pytorch
    
    Args:
        train_X(np.ndarray): The feature matrix, each row is a feature
        train_Y(np.ndarray): the label column vector
        model(LogisticRegressio): The model to be trained
        lr(float): The learning rate
        epoch_num(int): The number of epoch for training
        epoch_step(int): The number of step in one epoch
        optimizer(torch.optim.Optimizer): The optimizer used for training
        fig_path (str): The path to save the loss

    Returns:
        Dict[str, object]: The loss and F1 result
    """
    t = time.time()
    criterion = torch.nn.BCELoss()
    if not optimizer:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_record, F1_record = list(), list()
    train_Y = torch.from_numpy(train_Y).to(torch.float)
    i = 0
    indexes = np.arange(0, len(train_Y))
    test_indexes = np.arange(0, len(train_Y))
    batch_size = math.ceil(len(train_Y) / epoch_step)
    for epoch in tqdm.tqdm(range(epoch_num), desc="Training Epoch"):
        np.random.shuffle(indexes)
        np.random.shuffle(test_indexes)
        for step in tqdm.tqdm(
            range(epoch_step), desc="Training Step in Epoch {}".format(epoch)
        ):
            index = indexes[step * batch_size : (step + 1) * batch_size]
            batch_X = to_sparse_tensor(train_X[index])
            batch_Y = train_Y[index]
            predict = model(batch_X)
            optimizer.zero_grad()
            loss = criterion(predict, batch_Y)
            loss.backward()
            optimizer.step()
            test_index = test_indexes[step * batch_size : (step + 1) * batch_size]
            loss_record.append(float(loss.data))
            F1_record.append(evaluate(model, train_X[test_index], train_Y[test_index].detach().numpy().flatten())["F1"])
            i += 1
    logging.info(
        "Complete training for PyTorch version with {}s, the working set size is {} samples".format(
            round(time.time() - t, 2), epoch_num * epoch_step * batch_size
        )
    )
    if fig_path:
        df = pd.DataFrame(loss_record, columns=["loss"]).reset_index()
        chart = alt.Chart(df).mark_line().encode(y="loss:Q", x="index:Q",)
        chart.save(fig_path)
    return {"loss": loss_record, "F1": F1_record}


if __name__ == "__main__":
    # load the data
    train_df = load_df("train.csv")
    dev_df = load_df("dev.csv")

    # construct the vocabulary set and transform the input matrix
    processor = Preprocessor()
    matrix = processor.buildVocabulary(train_df)
    train_X = processor.buildMatrix(train_df)
    train_Y = train_df["party_affiliation"].values
    # build the model and train
    TorchModel = LogisticRegression(processor)
    train(train_X, train_Y, TorchModel, lr=5e-2, epoch_num=10, epoch_step=1000)
    # train(train_X, train_Y, TorchModel, lr=5e-3, epoch_num=1, epoch_step=1000)
    score = evaluate(TorchModel, dev_df, dev_df["party_affiliation"].values)

    # # save the model
    # torch.save(
    #     model.state_dict(),
    #     "/home/panda/models/TorchLR{}.pkl".format(round(score["F1"], 4) * 10000),
    # )
    logger.info(
        "TorchLR training completed, the score is {}".format(score)
    )
