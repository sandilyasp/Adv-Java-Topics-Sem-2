import numpy as np
import os
import pandas as pd
import random
import time

# Seed the random number generator with the current time
random.seed(time.time())

class StockDataSet(object):
    def __init__(self,
                 stock_sym,
                 input_size=1,
                 num_steps=30,
                 test_ratio=0.1,
                 normalized=True,
                 close_price_only=True):
        """
        Initializes the StockDataSet object with the given parameters.
        
        Args:
        stock_sym (str): Stock symbol.
        input_size (int): Number of input features.
        num_steps (int): Number of time steps.
        test_ratio (float): Ratio of the test set.
        normalized (bool): Whether to normalize the data.
        close_price_only (bool): Whether to use only close prices.
        """
        self.stock_sym = stock_sym
        self.input_size = input_size
        self.num_steps = num_steps
        self.test_ratio = test_ratio
        self.close_price_only = close_price_only
        self.normalized = normalized

        # Read the CSV file for the given stock symbol
        raw_df = pd.read_csv(os.path.join("data", "%s.csv" % stock_sym))

        # Use only the close prices if specified
        if close_price_only:
            self.raw_seq = raw_df['Close'].tolist()
        else:
            # Combine open and close prices into one sequence
            self.raw_seq = [price for tup in raw_df[['Open', 'Close']].values for price in tup]

        self.raw_seq = np.array(self.raw_seq)
        # Prepare the data for training and testing
        self.train_X, self.train_y, self.test_X, self.test_y = self._prepare_data(self.raw_seq)

    def info(self):
        """
        Returns information about the dataset.
        
        Returns:
        str: Information about the dataset.
        """
        return "StockDataSet [%s] train: %d test: %d" % (
            self.stock_sym, len(self.train_X), len(self.test_y))

    def _prepare_data(self, seq):
        """
        Prepares the data by normalizing it, creating input-output pairs,
        and splitting it into training and testing sets.
        
        Args:
        seq (array): The input sequence.
        
        Returns:
        tuple: Training and testing data (train_X, train_y, test_X, test_y).
        """
        # Split into items of input_size
        seq = [np.array(seq[i * self.input_size: (i + 1) * self.input_size])
               for i in range(len(seq) // self.input_size)]

        if self.normalized:
            # Normalize the data
            seq = [seq[0] / seq[0][0] - 1.0] + [
                curr / seq[i][-1] - 1.0 for i, curr in enumerate(seq[1:])]

        # Split into groups of num_steps
        X = np.array([seq[i: i + self.num_steps] for i in range(len(seq) - self.num_steps)])
        y = np.array([seq[i + self.num_steps] for i in range(len(seq) - self.num_steps)])

        # Calculate the size of the training set
        train_size = int(len(X) * (1.0 - self.test_ratio))
        # Split into training and testing sets
        train_X, test_X = X[:train_size], X[train_size:]
        train_y, test_y = y[:train_size], y[train_size:]
        return train_X, train_y, test_X, test_y

    def generate_one_epoch(self, batch_size):
        """
        Generates one epoch of data.
        
        Args:
        batch_size (int): The size of each batch.
        
        Yields:
        tuple: A batch of training data (batch_X, batch_y).
        """
        num_batches = int(len(self.train_X)) // batch_size
        if batch_size * num_batches < len(self.train_X):
            num_batches += 1

        # Shuffle the batch indices
        batch_indices = list(range(num_batches))
        random.shuffle(batch_indices)
        for j in batch_indices:
            batch_X = self.train_X[j * batch_size: (j + 1) * batch_size]
            batch_y = self.train_y[j * batch_size: (j + 1) * batch_size]
            # Ensure that all batches have the correct number of time steps
            assert set(map(len, batch_X)) == {self.num_steps}
            yield batch_X, batch_y
