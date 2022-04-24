from collections import Counter
from scipy import sparse
from typing import Callable, Dict, List, Union
import numpy as np
import pandas as pd
import re
import torch
from torch import nn
import tqdm
STOPWORDS = {
    "",
    "a",
    "about",
    "above",
    "all",
    "am",
    "an",
    "and",
    "are",
    "as",
    "at",
    "back",
    "be",
    "believe",
    "been",
    "believe" "below",
    "by",
    "click",
    "donate",
    "donation",
    "email",
    "emails",
    "error",
    "fewer",
    "for",
    "from",
    "he",
    "help",
    "here",
    "hi",
    "his",
    "how",
    "if",
    "image",
    "images",
    "in",
    "is",
    "it",
    "ll",
    "like",
    "longer",
    "me",
    "message",
    "of",
    "off",
    "on",
    "or",
    "our",
    "please",
    "re",
    "receive",
    "received",
    "saved",
    "sincerely",
    "thank",
    "thanks" "that",
    "the",
    "then",
    "there",
    "this",
    "those",
    "to",
    "turn",
    "unsubscribe",
    "url",
    "us",
    "ve",
    "want",
    "will",
    "wish",
    "we",
    "what",
    "when",
    "with",
    "would",
    "you",
    "your",
}


class Tokenizer(object):
    def __init__(self, ngram: int = 0):
        """A tokenizer to tokenize the document
        
        Args:
            ngram (int): The tokenize level, 0 means use a raw tokenize
        """
        self.ngram = ngram

    def __call__(self, doc: str) -> List[str]:
        """Token the document
        
        Args:
            doc (str): The string document to be tokenized

        Returns:
            List[str]: A list of token strings
        """
        if self.ngram == 0:
            return doc.split(" ")
        ret = list()
        words = ["" for i in range(self.ngram)]

        for token in doc.split(" "):
            if token in {"not", "never", "no"} or token.endswith(r"n't"):
                token = "not"
            else:
                token = re.sub("[^\w<>]", "", token)
                if len(token) == 1 or not token.isalpha() or token in STOPWORDS:
                    continue
            words.pop(0)
            words.append(token)
            ret.append("_".join(filter(lambda x: x != "", words)))
        return ret


def to_sparse_tensor(matrix: sparse.csr.csr_matrix) -> torch.Tensor:
    """Transform a sparse matrix into a tensor

    Args:
        matrix (sparse.csr.csr_matrix): A row-major sparse matrix

    Returns:
        (torch.Tensor): The Tensor representation of the sparse matrix
    """
    coo = matrix.tocoo()
    return torch.sparse_coo_tensor(
        np.mat([coo.row, coo.col]), coo.data, size=matrix.shape
    ).to(torch.float)


def load_df(path: str) -> pd.DataFrame:
    """Load the dataframe and do some preprocessing
    
    Args:
        path(str): The file path

    Returns:
        (pd.DataFrame): The processed dataframe
    """
    df = pd.read_csv(path)
    df["email_text"] = df["email_text"].str.lower()
    if "party_affiliation" in df.columns:
        df["party_affiliation"] = (
            (df["party_affiliation"] == "Democratic Party").astype(int).values
        )
    return df


def merge_df(feature_df: pd.DataFrame, predict: pd.Series) -> pd.DataFrame:
    """Merge the output of feature df and prediction
    
    Args:
        feature_df(pd.DataFrame): The feature dataframe
        predict(pd.Series): The prediction of the label

    Return:
        (pd.DataFrame): The dataframe to be exported
    """

    df = feature_df[["uid_email"]].copy()
    df["party_affiliation"] = predict.apply(
        lambda x: "Democratic Party" if x > 0.5 else "Republican Party"
    )
    return df.set_index("uid_email")


class Preprocessor(object):
    def __init__(
        self,
        tokenizer: Callable = Tokenizer(2),
        voc: Dict[str, int] = None,
        min_frequency: int = 10,
    ):
        """A class to build the vocabulary and transform matrix based on it

        Attributes:
            voc(Dict[str, int]): The vocabulary set
            tokenizer(Callable): The function to tokenize the document
            min_frequency(int): The min frequency of one word
        """
        self.voc = voc if voc else dict()
        self.tokenizer = tokenizer
        self.min_frequency = min_frequency

    def buildVocabulary(self, train_df: pd.DataFrame):
        """Build/expend the vocabulary from the dataframe
        
        Args:
            train_df(pd.DataFrame): The dataframe used to build the vocabulary
        """
        ct = Counter()
        index = 0
        for row in tqdm.tqdm(train_df["email_text"], desc="Build Voc"):
            ct.update(Counter(self.tokenizer(row)))

        for k in sorted(ct.keys()):
            if ct[k] >= self.min_frequency:
                self.voc[k] = index
                index += 1

    def buildMatrix(self, df: pd.DataFrame) -> sparse.csr.csr_matrix:
        """Build the document matrix from the dataframe
        
        Args:
            df(pd.DataFrame): The document to be built from

        Returns:
            (sparse.csr.csr_matrix): The rowwise compressed sparse matrix
        """
        if not isinstance(df, pd.DataFrame):
            return sparse.csr_matrix(df)
        data = list()
        indptr = [0]
        indices = list()
        for i, row in tqdm.tqdm(enumerate(df["email_text"]), desc="Build Matrix"):
            arr = np.zeros(len(self.voc) + 1)
            words = self.tokenizer(row)
            for word in words:
                if word in self.voc.keys():
                    data.append(1)
                    indices.append(self.voc[word])
            data.append(1)
            indices.append(len(self.voc))
            indptr.append(len(indices))
        matrix = sparse.csr_matrix((data, indices, indptr), dtype=int)
        return matrix


def predict(model: object, features: Union[pd.DataFrame, np.ndarray]):
    """Make prediction based on the model and input
        
    Args:
        model(object): The model to be used for prediction
        features(Union[pd.DataFrame, np.ndarray]): The feature dataframe
    """
    if isinstance(model, nn.Module):
        if not isinstance(features, torch.Tensor):
            features = to_sparse_tensor(model.processor.buildMatrix(features))
        output = model(features).detach().numpy().flatten()
    else:
        output = model(features)
    return pd.Series(output)


def evaluate(
    model: object, features: Union[pd.DataFrame, np.ndarray], label: pd.Series
) -> Dict[str, float]:
    """Output the performance of the result
    
    Args:
        model(object): The model to be used for prediction
        features(Union[pd.DataFrame, np.ndarray]): The feature dataframe
        label(pd.Series): The label output

    Return:
        (Dict[str, float]): The performance of the model
    """
    prediction = predict(model, features)
    output = pd.DataFrame({"Prediction": prediction > 0.5, "Label": label}).astype(bool)
    TP = len(output[(output["Prediction"]) & (output["Label"])]) / len(output)
    TN = len(output[(~output["Prediction"]) & (~output["Label"])]) / len(output)
    FP = len(output[(output["Prediction"]) & (~output["Label"])]) / len(output)
    FN = len(output[(~output["Prediction"]) & (output["Label"])]) / len(output)
    accuracy = len(output[(output["Prediction"]) == (output["Label"])]) / len(output)
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    F1 = (
        (2 * precision * recall) / (precision + recall)
        if precision + recall != 0
        else 0
    )
    return {"Precision": precision, "Recall": recall, "F1": F1, "Accuracy": accuracy}
