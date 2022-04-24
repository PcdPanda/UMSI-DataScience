import logging
import pickle
import time
from typing import List, Callable, Dict
from Utils import *
import altair as alt
import numpy as np
import pandas as pd
import tqdm

logger = logging.getLogger("Numpy Logistic Regression Logger")
logging.basicConfig(level=logging.DEBUG)


def sigmoid(X: np.ndarray) -> np.ndarray:
    """Calculte the sigmoid, notice the overflow
    
    Args:
        X(np.ndarray): The input

    Returns:
        (np.ndarray): The sigmoid output
    """
    return 1 / (1 + np.exp(-X))


class NumpyLR(object):
    def __init__(self, processor: Preprocessor):
        """A class for logistic regression using numpy
        
        Args:
            processor (Preprocessor): A processor with vocabular set 

        Attributes:
            processor (Preprocessor):
            beta (np.ndarray): A column vector saving the parameters
        """
        self.processor = processor
        self.beta = np.zeros(len(self.processor.voc) + 1)

    def log_likelihood(self, Y: np.ndarray, X: np.ndarray) -> float:
        """Calculate the log likelihood
    
        Args:
            Y(np.ndarray): The label column vector
            X(np.ndarray): The feature matrix, each column  is a feature

        Returns:
            (float): The likelihood of the logistic regression
        """
        beta_X = X.dot(self.beta).T
        ret = Y.dot(beta_X) - np.log(1 + np.exp(beta_X)).dot(np.ones(len(Y)))
        return ret

    @staticmethod
    def gradient(
        train_X: np.ndarray, train_Y: np.ndarray, predict_Y: np.ndarray
    ) -> np.ndarray:
        """Calculate the gradient of the log likelihood
        
        Args:
            X(np.ndarray): The feature matrix, each row is a feature
            Y(np.ndarray): the label column vector
            Y_predict(np.ndarray): The prediction column vector

        Returns:
            (np.ndarray): The gradient of the log_likelihood function
        """
        return train_X.T.dot(train_Y - predict_Y)

    def logistic_regression(
        self,
        train_X: np.ndarray,
        train_Y: np.ndarray,
        lr: float = 5e-5,
        num_step: int = 1e3,
        batch_size: int = 1,
        fig_path: str = "",
    ) -> Dict[str, object]:
        """Run the logistic regression using SGD
    
        Args:
            train_X(np.ndarray): The feature matrix, each row is a feature
            train_Y(np.ndarray): the label column vector
            lr(float): How beta change each step
            num_step(int): The step number for updating
            batch_size(int): The batch_size to calculate the result
            path(str): The path to save the likelihood result

        Returns:
            [str, object]: The likelihood result
        """
        likelihood = list()
        for i in tqdm.tqdm(range(int(num_step))):
            index = np.random.choice(len(train_Y), batch_size, replace=False)
            batch_X, batch_Y = train_X[index], train_Y[index]
            predict_Y = sigmoid(batch_X.dot(self.beta))
            grad = self.gradient(batch_X, batch_Y, predict_Y)
            if not i % (num_step / 10):
                likelihood.append(self.log_likelihood(batch_Y, batch_X) / len(train_Y))
            self.beta += lr * grad
        if fig_path:
            df = pd.DataFrame(likelihood, columns=["likelihood"]).reset_index()
            chart = alt.Chart(df).mark_line().encode(y="likelihood:Q", x="index:Q",)
            chart.save(fig_path)
        return {"likelihood": likelihood}

    def __call__(self, df: pd.DataFrame):
        """Make prediction based on the input
        
        Args:
            df(pd.DataFrame): The input for prediction

        Returns:
            (np.ndarray): The output result of the prediction
        """
        features = self.processor.buildMatrix(df)
        return features.dot(self.beta)

    def save(self, path: str):
        """Using pickle to save the model for future usage
        
        Args:
            path(str): The path to save to the model
        """
        with open(path, "wb") as f:
            f.write(pickle.dumps(self))

    @classmethod
    def load(cls, path: str):
        """Using pickle to load the model 

        Args:
            path(str): The path to load to the model

        Returns:
            (NumpyLR): The model
        """
        with open(path, "rb") as f:
            return pickle.loads(f.read())

    def get_token_weights(self):
        """Sort the words by their weights
        
        Returns:
            (List[float, str]): Sorted tokens list by their weight
        """
        pri = dict()
        for k, v in self.processor.voc.items():
            pri[v] = k
        return sorted(
            [(val, pri[i]) for i, val in enumerate(self.beta[:-1])],
            key=lambda x: -abs(x[0]),
        )


def train(
    train_X: np.ndarray,
    train_Y: np.ndarray,
    model: NumpyLR,
    lr: float = 5e-5,
    num_step: int = 1e3,
    batch_size: int = 1,
    fig_path: str = "",
) -> NumpyLR:
    """Train the model through pytorch
    
    Args:
        train_X(np.ndarray): The feature matrix, each row is a feature
        train_Y(np.ndarray): the label column vector
        model(NumpyLR): The model to be trained
        lr(float): The learning rate
        num_step(int): The number of epoch for training
        fig_path (str): The path to save the loss

    Returns:
        (NumpyLR): The model after training
    """
    t = time.time()
    likelihood = model.logistic_regression(
        train_X, train_Y, lr, num_step, batch_size, fig_path
    )
    logger.info(
        "Complete training for NumPy version with {}s, the working set size is {} samples".format(
            round(time.time() - t, 2), num_step * batch_size
        )
    )
    return model

if __name__ == "__main__":
    # load the data
    train_df = load_df("train.csv")
    dev_df = load_df("dev.csv")

    # construct the vocabulary set and transform the input matrix
    processor = Preprocessor()
    processor.buildVocabulary(train_df)
    train_X = processor.buildMatrix(train_df)
    train_Y = train_df["party_affiliation"].values

    # build the model and train
    NumpyModel = NumpyLR(processor)
    train(train_X, train_Y, NumpyModel, lr=1e-4, num_step=1e3, batch_size=10000)
    score = evaluate(NumpyModel, dev_df, dev_df["party_affiliation"].values)
    # 
    logger.info(
        "Numpy training completed, the score is {}".format(score)
    )