import numpy as np
import os
import random
import re
import shutil
import time
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.contrib.tensorboard.plugins import projector

class LstmRNN(object):
    def __init__(self, sess, stock_count,
                 lstm_size=128,
                 num_layers=1,
                 num_steps=30,
                 input_size=1,
                 embed_size=None,
                 logs_dir="logs",
                 plots_dir="images"):
        """
        Construct a RNN model using LSTM cell.

        Args:
            sess: TensorFlow session.
            stock_count (int): Number of stocks we are going to train with.
            lstm_size (int): Size of LSTM hidden units.
            num_layers (int): Number of LSTM cell layers.
            num_steps (int): Number of time steps.
            input_size (int): Dimension of input features.
            embed_size (int): Length of embedding vector, only used when stock_count > 1.
            logs_dir (str): Directory to save logs.
            plots_dir (str): Directory to save plots.
        """
        self.sess = sess
        self.stock_count = stock_count

        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.num_steps = num_steps
        self.input_size = input_size

        self.use_embed = (embed_size is not None) and (embed_size > 0)
        self.embed_size = embed_size or -1

        self.logs_dir = logs_dir
        self.plots_dir = plots_dir

        self.build_graph()

    def build_graph(self):
        """
        Build the computation graph for the LSTM RNN model.
        """
        # Define placeholders for input, output, learning rate, and keep probability.
        self.learning_rate = tf.placeholder(tf.float32, None, name="learning_rate")
        self.keep_prob = tf.placeholder(tf.float32, None, name="keep_prob")

        # Placeholder for stock symbols.
        self.symbols = tf.placeholder(tf.int32, [None, 1], name='stock_labels')

        # Placeholder for inputs and targets.
        self.inputs = tf.placeholder(tf.float32, [None, self.num_steps, self.input_size], name="inputs")
        self.targets = tf.placeholder(tf.float32, [None, self.input_size], name="targets")

        # Function to create an LSTM cell with dropout.
        def _create_one_cell():
            lstm_cell = tf.contrib.rnn.LSTMCell(self.lstm_size, state_is_tuple=True)
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob)
            return lstm_cell

        # Create a multi-layer LSTM cell if num_layers > 1, else create a single LSTM cell.
        cell = tf.contrib.rnn.MultiRNNCell(
            [_create_one_cell() for _ in range(self.num_layers)],
            state_is_tuple=True
        ) if self.num_layers > 1 else _create_one_cell()

        # If embedding is used, create embedding matrix and concatenate it with inputs.
        if self.embed_size > 0 and self.stock_count > 1:
            self.embed_matrix = tf.Variable(
                tf.random_uniform([self.stock_count, self.embed_size], -1.0, 1.0),
                name="embed_matrix"
            )
            
            # Tile stock labels to match the number of time steps.
            stacked_symbols = tf.tile(self.symbols, [1, self.num_steps], name='stacked_stock_labels')
            stacked_embeds = tf.nn.embedding_lookup(self.embed_matrix, stacked_symbols)

            # Concatenate input features with embedding.
            self.inputs_with_embed = tf.concat([self.inputs, stacked_embeds], axis=2, name="inputs_with_embed")
            self.embed_matrix_summ = tf.summary.histogram("embed_matrix", self.embed_matrix)
        else:
            self.inputs_with_embed = tf.identity(self.inputs)
            self.embed_matrix_summ = None

        print("inputs.shape:", self.inputs.shape)
        print("inputs_with_embed.shape:", self.inputs_with_embed.shape)

        # Run the LSTM cell on the input data.
        val, state_ = tf.nn.dynamic_rnn(cell, self.inputs_with_embed, dtype=tf.float32, scope="dynamic_rnn")

        # Transpose the output to match the expected shape.
        val = tf.transpose(val, [1, 0, 2])

        # Get the last output of the LSTM for prediction.
        last = tf.gather(val, int(val.get_shape()[0]) - 1, name="lstm_state")
        ws = tf.Variable(tf.truncated_normal([self.lstm_size, self.input_size]), name="w")
        bias = tf.Variable(tf.constant(0.1, shape=[self.input_size]), name="b")
        self.pred = tf.matmul(last, ws) + bias

        # Summaries for TensorBoard.
        self.last_sum = tf.summary.histogram("lstm_state", last)
        self.w_sum = tf.summary.histogram("w", ws)
        self.b_sum = tf.summary.histogram("b", bias)
        self.pred_summ = tf.summary.histogram("pred", self.pred)

        # Define the loss function (mean squared error) and the optimizer.
        self.loss = tf.reduce_mean(tf.square(self.pred - self.targets), name="loss_mse_train")
        self.optim = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss, name="rmsprop_optim")

        # Separate loss for testing.
        self.loss_test = tf.reduce_mean(tf.square(self.pred - self.targets), name="loss_mse_test")

        # Summaries for TensorBoard.
        self.loss_sum = tf.summary.scalar("loss_mse_train", self.loss)
        self.loss_test_sum = tf.summary.scalar("loss_mse_test", self.loss_test)
        self.learning_rate_sum = tf.summary.scalar("learning_rate", self.learning_rate)

        # Saver to save and restore model checkpoints.
        self.t_vars = tf.trainable_variables()
        self.saver = tf.train.Saver()

    def train(self, dataset_list, config):
        """
        Train the LSTM RNN model.

        Args:
            dataset_list (list of StockDataSet): List of dataset objects.
            config (tf.app.flags.FLAGS): Training configuration.
        """
        assert len(dataset_list) > 0
        self.merged_sum = tf.summary.merge_all()

        # Set up the logs folder.
        self.writer = tf.summary.FileWriter(os.path.join("./logs", self.model_name))
        self.writer.add_graph(self.sess.graph)

        if self.use_embed:
            # Set up embedding visualization.
            projector_config = projector.ProjectorConfig()
            added_embed = projector_config.embeddings.add()
            added_embed.tensor_name = self.embed_matrix.name
            shutil.copyfile(os.path.join(self.logs_dir, "metadata.tsv"),
                            os.path.join(self.model_logs_dir, "metadata.tsv"))
            added_embed.metadata_path = "metadata.tsv"
            projector.visualize_embeddings(self.writer, projector_config)

        tf.global_variables_initializer().run()

        # Merge test data from different stocks.
        merged_test_X = []
        merged_test_y = []
        merged_test_labels = []

        for label_, d_ in enumerate(dataset_list):
            merged_test_X += list(d_.test_X)
            merged_test_y += list(d_.test_y)
            merged_test_labels += [[label_]] * len(d_.test_X)

        merged_test_X = np.array(merged_test_X)
        merged_test_y = np.array(merged_test_y)
        merged_test_labels = np.array(merged_test_labels)

        print("len(merged_test_X) =", len(merged_test_X))
        print("len(merged_test_y) =", len(merged_test_y))
        print("len(merged_test_labels) =", len(merged_test_labels))

        test_data_feed = {
            self.learning_rate: 0.0,
            self.keep_prob: 1.0,
            self.inputs: merged_test_X,
            self.targets: merged_test_y,
            self.symbols: merged_test_labels,
        }

        global_step = 0

        # Calculate the number of batches for training.
        num_batches = sum(len(d_.train_X) for d_ in dataset_list) // config.batch_size
        random.seed(time.time())

        # Select samples for plotting.
        sample_labels = range(min(config.sample_size, len(dataset_list)))
        sample_indices = {}
        for l in sample_labels:
            sym = dataset_list[l].stock_sym
            target_indices = np.array([
                i for i, sym_label in enumerate(merged_test_labels)
                if sym_label[0] == l])
            sample_indices[sym] = target_indices
        print(sample_indices)

        print("Start training for stocks:", [d.stock_sym for d in dataset_list])
        for epoch in range(config.max_epoch):
            epoch_step = 0
            learning_rate = config.init_learning_rate * (
                config.learning_rate_decay ** max(float(epoch + 1 - config.init_epoch), 0.0)
            )

            for label_, d_ in enumerate(dataset_list):
                for batch_X, batch_y in d_.generate_one_epoch(config.batch_size):
                    global_step += 1
                    epoch_step += 1
                    batch_labels = np.array([[label_]] * len(batch_X))
                    train_data_feed = {
                        self.learning_rate: learning_rate,
                        self.keep_prob: config.keep_prob,
                        self.inputs: batch_X,
                        self.targets: batch_y,
                        self.symbols: batch_labels,
                    }
                    train_loss, _, train_merged_sum = self.sess.run(
                        [self.loss, self.optim, self.merged_sum], train_data_feed)
                    self.writer.add_summary(train_merged_sum, global_step=global_step)

                    if np.mod(global_step, len(dataset_list) * 200 / config.input_size) == 1:
                        test_loss, test_pred = self.sess.run([self.loss_test, self.pred], test_data_feed)

                        print("Step:%d [Epoch:%d] [Learning rate: %.6f] train_loss:%.6f test_loss:%.6f" % (
                            global_step, epoch, learning_rate, train_loss, test_loss))

                        # Plot samples.
                        for sample_sym, indices in sample_indices.items():
                            image_path = os.path.join(self.model_plots_dir, "{}_epoch{:02d}_step{:04d}.png".format(
                                sample_sym, epoch, epoch_step))
                            sample_preds = test_pred[indices]
                            sample_truth = merged_test_y[indices]
                            self.plot_samples(sample_preds, sample_truth, image_path, stock_sym=sample_sym)

                        self.save(global_step)

        final_pred, final_loss = self.sess.run([self.pred, self.loss], test_data_feed)

        # Save the final model.
        self.save(global_step)
        return final_pred

    @property
    def model_name(self):
        name = "stock_rnn_lstm%d_step%d_input%d" % (
            self.lstm_size, self.num_steps, self.input_size)

        if self.embed_size > 0:
            name += "_embed%d" % self.embed_size

        return name

    @property
    def model_logs_dir(self):
        model_logs_dir = os.path.join(self.logs_dir, self.model_name)
        if not os.path.exists(model_logs_dir):
            os.makedirs(model_logs_dir)
        return model_logs_dir

    @property
    def model_plots_dir(self):
        model_plots_dir = os.path.join(self.plots_dir, self.model_name)
        if not os.path.exists(model_plots_dir):
            os.makedirs(model_plots_dir)
        return model_plots_dir

    def save(self, step):
        model_name = self.model_name + ".model"
        self.saver.save(
            self.sess,
            os.path.join(self.model_logs_dir, model_name),
            global_step=step
        )

    def load(self):
        """
        Load the latest checkpoint from the model directory.
        """
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(self.model_logs_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(self.model_logs_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def plot_samples(self, preds, targets, figname, stock_sym=None, multiplier=5):
        """
        Plot predictions vs targets for a stock.

        Args:
            preds: Predictions from the model.
            targets: True target values.
            figname: File name to save the plot.
            stock_sym: Stock symbol (optional).
            multiplier: Multiplier to scale predictions (default is 5).
        """
        def _flatten(seq):
            return np.array([x for y in seq for x in y])

        truths = _flatten(targets)[-200:]
        preds = (_flatten(preds) * multiplier)[-200:]
        days = range(len(truths))[-200:]

        plt.figure(figsize=(12, 6))
        plt.plot(days, truths, label='truth')
        plt.plot(days, preds, label='pred')
        plt.legend(loc='upper left', frameon=False)
        plt.xlabel("day")
        plt.ylabel("normalized price")
        plt.ylim((min(truths), max(truths)))
        plt.grid(ls='--')

        if stock_sym:
            plt.title(stock_sym + " | Last %d days in test" % len(truths))

        plt.savefig(figname, format='png', bbox_inches='tight', transparent=True)
        plt.close()
