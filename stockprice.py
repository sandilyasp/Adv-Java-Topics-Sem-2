import os
import pandas as pd
import pprint

import tensorflow as tf
import tensorflow.contrib.slim as slim

from data_model import StockDataSet
from model_rnn import LstmRNN

# Define flags for various hyperparameters and settings
flags = tf.app.flags
flags.DEFINE_integer("stock_count", 100, "Number of stocks to use [100]")
flags.DEFINE_integer("input_size", 1, "Input size for the model [1]")
flags.DEFINE_integer("num_steps", 30, "Number of time steps [30]")
flags.DEFINE_integer("num_layers", 1, "Number of LSTM layers [1]")
flags.DEFINE_integer("lstm_size", 128, "Size of one LSTM cell [128]")
flags.DEFINE_integer("batch_size", 64, "The size of the batch [64]")
flags.DEFINE_float("keep_prob", 0.8, "Keep probability for dropout [0.8]")
flags.DEFINE_float("init_learning_rate", 0.001, "Initial learning rate [0.001]")
flags.DEFINE_float("learning_rate_decay", 0.99, "Decay rate for the learning rate [0.99]")
flags.DEFINE_integer("init_epoch", 5, "Number of epochs considered as early stage [5]")
flags.DEFINE_integer("max_epoch", 50, "Total number of training epochs [50]")
flags.DEFINE_integer("embed_size", None, "Size of embedding vector if provided [None]")
flags.DEFINE_string("stock_symbol", None, "Target stock symbol [None]")
flags.DEFINE_integer("sample_size", 4, "Number of stocks to plot during training [4]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")

FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()

# Create logs directory if it doesn't exist
if not os.path.exists("logs"):
    os.mkdir("logs")

# Function to show all trainable variables
def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

# Function to load S&P 500 stock data
def load_sp500(input_size, num_steps, k=None, target_symbol=None, test_ratio=0.05):
    if target_symbol is not None:
        return [
            StockDataSet(
                target_symbol,
                input_size=input_size,
                num_steps=num_steps,
                test_ratio=test_ratio)
        ]

    # Load metadata of S&P 500 stocks
    info = pd.read_csv("data/constituents-financials.csv")
    info = info.rename(columns={col: col.lower().replace(' ', '_') for col in info.columns})
    info['file_exists'] = info['symbol'].map(lambda x: os.path.exists("data/{}.csv".format(x)))
    print(info['file_exists'].value_counts().to_dict())

    info = info[info['file_exists'] == True].reset_index(drop=True)
    info = info.sort_values('market_cap', ascending=False).reset_index(drop=True)

    if k is not None:
        info = info.head(k)

    print("Head of S&P 500 info:\n", info.head())

    # Generate embedding meta file
    info[['symbol', 'sector']].to_csv(os.path.join("logs/metadata.tsv"), sep='\t', index=False)

    return [
        StockDataSet(row['symbol'],
                     input_size=input_size,
                     num_steps=num_steps,
                     test_ratio=0.05)
        for _, row in info.iterrows()]

# Main function to execute the model training/testing
def main(_):
    pp.pprint(flags.FLAGS.__flags)

    # Configure GPU settings
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:
        # Initialize the LSTM RNN model
        rnn_model = LstmRNN(
            sess,
            FLAGS.stock_count,
            lstm_size=FLAGS.lstm_size,
            num_layers=FLAGS.num_layers,
            num_steps=FLAGS.num_steps,
            input_size=FLAGS.input_size,
            embed_size=FLAGS.embed_size,
        )

        show_all_variables()

        # Load stock data
        stock_data_list = load_sp500(
            FLAGS.input_size,
            FLAGS.num_steps,
            k=FLAGS.stock_count,
            target_symbol=FLAGS.stock_symbol,
        )

        if FLAGS.train:
            # Train the model
            rnn_model.train(stock_data_list, FLAGS)
        else:
            # Test the model
            if not rnn_model.load()[0]:
                raise Exception("[!] Train a model first, then run test mode")

# Entry point of the script
if __name__ == '__main__':
    tf.app.run()
