import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from tqdm.auto import tqdm, trange
from collections import Counter
import random
from torch import optim
import json
import pandas as pd
import logging
from scipy.spatial.distance import cosine
from typing import *
import traceback

logger = logging.getLogger("Word2Vec")

with open("words.json", "r") as f:
    words = json.loads(f.read())
    synonyms = words["synonyms"]
    stopwords = words["stopwords"]
    prefixes = words["prefix"]
    suffixes = words["suffixes"]
    phrases = words["phrases"]


def UNKing(token):
    return "<UNK>"
    if " " in token:
        return token.lower()
    if not token.isalpha():
        return "<UNK_NUM>"
    elif token.upper() == token:
        return "<UNK_CAP>"
    token = token.lower()
    pre, suf = "", ""
    for i, prefix in enumerate(prefixes):
        if isinstance(prefix, list):
            prefix = tuple(prefix)
        if token.startswith(prefix):
            pre = i
            break
    for i, suffix in enumerate(suffixes):
        if isinstance(suffix, list):
            suffix = tuple(suffix)
        if token.startswith(suffix):
            suf = i
            break
    return "<UNK_{}_{}>".format(pre, suf)


class Corpus(object):
    def __init__(self):
        self.word_to_index = dict()  # word to unique-id
        self.index_to_word = dict()  # unique-id to word
        self.word_counts = Counter()
        self.negative_sampling_table = list()
        self.full_token_sequence_as_ids = list()
        self.training_data = list()

    def load_data(self, tokens: str, min_token_freq: int):
        """
        Reads the data from the specified file as long long sequence of text
        (ignoring line breaks) and populates the data structures of this
        word2vec object.
        """
        counter = Counter()
        self.word_counts.clear()
        all_tokens = tokens.copy()
        counter.update(all_tokens)
        for i, token in enumerate(tqdm(all_tokens)):
            if counter[token] < min_token_freq:
                all_tokens[i] = UNKing(token)
            else:
                if token.upper() != token:
                    token = token.lower()
                    if token in synonyms.keys():
                        p = list()
                        for syn in synonyms[token]:
                            p.append(1 / counter.get(syn, 1))
                        words = [token] + synonyms[token]
                        p = np.array([sum(p)] + p) / (2 * sum(p))
                        token = np.random.choice(words, p=p)
                all_tokens[i] = token
        self.word_counts.update(all_tokens)
        for i, word in enumerate(sorted(self.word_counts.keys())):
            self.word_to_index[word] = i
            self.index_to_word[i] = word
        prob = np.array(
            [
                self.word_counts[self.index_to_word[i]]
                for i in sorted(self.index_to_word.keys())
            ]
        ) / len(all_tokens)
        prob = (np.power((prob / 1e-3), 0.5) + 1) * 1e-3 / prob
        self.full_token_sequence_as_ids = [
            self.word_to_index[word] for word in all_tokens
        ]
        # subsampling
        self.full_token_sequence_as_ids = [
            index
            for index in self.full_token_sequence_as_ids
            if np.random.binomial(1, min(prob[index], 1))
        ]
        logger.info(
            "Loaded all data; saw %d tokens (%d unique)"
            % (len(self.full_token_sequence_as_ids), len(self.word_to_index))
        )

    def generate_negative_samples(self, cur_context_word_id, num_samples):
        """
        Randomly samples the specified number of negative samples from the lookup
        table and returns this list of IDs as a numpy array. As a performance
        improvement, avoid sampling a negative example that has the same ID as
        the current positive context word.
        """
        results = list()
        context_words = set(cur_context_word_id)
        while len(results) < num_samples:
            sample = int(np.random.choice(self.negative_sampling_table, 1)[0])
            if (
                sample not in cur_context_word_id
                and self.index_to_word[sample] not in stopwords
            ):
                results.append(sample)

        return np.array(results)

    def generate_training_data(
        self,
        window_size: int = 2,
        neg_num: int = 2,
        exp_power: float = 0.75,
        table_size: float = 1e6,
    ):
        prob = np.power(
            [
                self.word_counts[self.index_to_word[i]]
                for i in sorted(self.index_to_word.keys())
            ],
            exp_power,
        )
        prob = prob / prob.sum()
        self.negative_sampling_table = np.random.choice(
            len(prob), int(table_size), p=prob
        )
        logger.info("Generated sampling table")

        for i, index in enumerate(tqdm(self.full_token_sequence_as_ids)):
            pos_samples = list()
            if self.index_to_word[index].startswith(
                "<UNK_"
            ):  # don;t use unk as target words
                continue
            for offset in range(-window_size, window_size + 1):
                if offset == 0:
                    continue
                if (
                    i + offset < len(self.full_token_sequence_as_ids)
                    and i + offset >= 0
                ):
                    if (
                        self.index_to_word[
                            self.full_token_sequence_as_ids[i + offset]
                        ].lower()
                        in stopwords
                    ):
                        continue  # remove the stopwords
                    pos_samples.append(self.full_token_sequence_as_ids[i + offset])
            num_samples = 2 * window_size * (1 + neg_num) - len(pos_samples)
            neg_samples = self.generate_negative_samples(pos_samples, num_samples)
            samples = (
                np.append(pos_samples, neg_samples) if pos_samples else neg_samples
            )
            labels = np.append(
                np.ones(len(pos_samples), dtype=np.float32),
                np.zeros(len(neg_samples), dtype=np.float32),
            )
            self.training_data.append([index, samples, labels])


class Word2Vec(nn.Module):
    def __init__(self, vocab_size: int, embedding_size: int):
        super(Word2Vec, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.init_emb(init_range=0.5 / self.vocab_size)
        self.output_layer = nn.Sigmoid()

    def init_emb(self, init_range):
        self.context_embeddings = nn.Embedding(self.vocab_size, self.embedding_size)
        self.context_embeddings.weight.data.uniform_(-init_range, init_range)
        self.target_embeddings = nn.Embedding(self.vocab_size, self.embedding_size)
        self.target_embeddings.weight.data.uniform_(-init_range, init_range)

    def forward(self, target_word_id, context_word_ids):
        """ 
        Predicts whether each context word was actually in the context of the target word.
        The input is a tensor with a single target word's id and a tensor containing each
        of the context words' ids (this includes both positive and negative examples).
        """
        target_vec = self.context_embeddings(target_word_id).unsqueeze(1)
        context_vec = self.target_embeddings(context_word_ids)
        # random dropoff
        target_rand, context_rand = (
            torch.rand(target_vec.shape),
            torch.rand(context_vec.shape),
        )
        target_vec = torch.where(
            target_rand > 0.1, target_vec, torch.zeros(target_vec.shape)
        )
        context_vec = torch.where(
            context_rand > 0.1, context_vec, torch.zeros(context_rand.shape)
        )
        return self.output_layer(torch.sum(target_vec * context_vec, dim=-1))

    def save(self, path: str):
        torch.save(self.state_dict(), path)


class W2V(object):
    def __init__(self, model, word_to_index: Dict[str, int]):
        self.model = model
        self.word_to_index = word_to_index

    def get_vec(self, word: str):
        if word.upper() != word:
            word = word.lower()
        if word not in self.word_to_index.keys():
            word = UNKing(word)
        return (
            self.model.target_embeddings(torch.LongTensor([self.word_to_index[word]]))
            .squeeze()
            .detach()
            .numpy()
        )

    @staticmethod
    def compute_cosine(vec1, vec2):
        return 1 - abs(float(cosine(vec1, vec2)))

    def compute_cosine_similarity(self, word_one: str, word_two: str):
        """Computes the cosine similarity between the two words"""
        try:
            return self.compute_cosine(self.get_vec(word_one), self.get_vec(word_two))
        except Exception:
            logger.error(traceback.format_exc())
            return 0

    def get_neighbors(self, target_word: str):
        """ Finds the top 10 most similar words to a target word"""
        outputs = []
        for word, index in tqdm(
            self.word_to_index.items(), total=len(self.word_to_index)
        ):
            similarity = self.compute_cosine_similarity(target_word, word)
            result = {"word": word, "score": similarity}
            outputs.append(result)

        # Sort by highest scores
        neighbors = sorted(outputs, key=lambda o: o["score"], reverse=True)
        return neighbors[1:11]

    def train(
        self, corpus: Corpus, lr: float = 5e-5, batch_size: int = 512, epochs: int = 1
    ):
        dataloader = DataLoader(corpus.training_data, batch_size=512, shuffle=True)
        criterion = torch.nn.BCELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: max(0.995 ** step, 5e-3)
        )
        loss_sum, bias = 0, 0
        for epoch in trange(epochs):
            for step, data in enumerate(tqdm(dataloader)):
                optimizer.zero_grad()
                target_ids, context_ids, labels = data
                output = self.model(target_ids, context_ids)
                loss = criterion(torch.squeeze(output), torch.squeeze(labels))
                loss.backward()
                loss_sum += float(loss.detach().numpy())
                optimizer.step()
                if not step % 100:  # plot the loss sum every 100 step
                    loss_sum = 0
                    scheduler.step()
        self.word_to_index = corpus.word_to_index
        return self.model
