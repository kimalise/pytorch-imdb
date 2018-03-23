from model.simple_gru import SimpleGru
from imdb_dataset import load_imdb_data
from imdb_dataset import IMDBDataset
from model.simple_gru import create_variable
import torch
import torch.nn as nn
import numpy as np
import math
import pickle

class ModelRunner:
    def __init__(self, params):

        x_train, y_train, x_dev, y_dev, x_test, y_test, word2idx, idx2word = load_imdb_data(sample=params['sample'])

        self.sample = params['sample']
        self.model_name = params['model']
        self.mode = params['mode']
        self.vocab_size = len(word2idx)
        self.batch_size = params['batch_size']
        self.embedding_size = 100
        self.hidden_size = 128
        self.output_size = 2
        self.epoch = params['epoch']
        self.learning_rate = params['learning_rate']

        self.train_dataset = IMDBDataset(x_train, y_train, word2idx, idx2word, batch_size=self.batch_size, shuffle=True)
        self.dev_dataset = IMDBDataset(x_dev, y_dev, word2idx, idx2word, batch_size=len(x_dev), shuffle=False)
        self.test_dataset = IMDBDataset(x_test, y_test, word2idx, idx2word, batch_size=len(x_test), shuffle=False)

        if params['model'] == 'simple-gru':
            self.model = SimpleGru(vocab_size=self.vocab_size, embedding_size=self.embedding_size, hidden_size=self.hidden_size,
                                   output_size=self.output_size)

        if torch.cuda.is_available():
            self.model.cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def train(self):

        for epoch in range(1, self.epoch + 1):
            step = 1
            total_loss = 0.0

            total_batches = math.ceil(len(self.train_dataset.inputs) / self.batch_size)

            for seq_padded, seq_lengths, targets in self.train_dataset.batches():
                seq_padded = create_variable(torch.LongTensor(seq_padded))
                seq_lengths = create_variable(torch.LongTensor(seq_lengths))
                targets = create_variable(torch.LongTensor(targets))

                output = self.model(seq_padded, seq_lengths)

                # targets = create_variable(targets)
                loss = self.criterion(output, targets)
                total_loss += loss.data[0]

                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()

                # avg_loss = total_loss / (step * self.batch_size)
                # print("epoch ", epoch, "    step: ", step,  " / ", total_batches, "    loss: ", loss.data[0])
                print("epoch: {}/{}, step: {}/{}, loss: {:.3f}".format(epoch, self.epoch, step, total_batches, loss.data[0]))

                if step % 100 == 0:
                    self.validate()

                step += 1

        self.test()

        self.save_model()

        return

    def validate(self):
        return self._evaluate(test=False)

    def test(self):
        return self._evaluate(test=True)

    def _evaluate(self, test=False):

        if test:
            dataset = self.test_dataset
        else:
            dataset = self.dev_dataset

        step = 1
        all_targets = np.array([])
        all_predict = np.array([])
        for seq_padded, seq_lengths, targets in dataset.batches():
            seq_padded = create_variable(torch.LongTensor(seq_padded))
            seq_lengths = create_variable(torch.LongTensor(seq_lengths))
            # targets = create_variable(torch.LongTensor(targets))

            output = self.model(seq_padded, seq_lengths)
            output = output.data.cpu().numpy()
            pred = np.argmax(output, axis=1)

            all_targets = np.append(all_targets, targets)
            all_predict = np.append(all_predict, pred)

            step += 1

        # calculate accuracy
        acc = sum([1 if p == y else 0 for p, y in zip(all_predict, all_targets)]) / len(all_targets)
        print("accuracy: ", acc)
        return acc


    def save_model(self):
        dataset = "sample" if self.sample else "full"
        path = f"saved_models/{dataset}_{self.model_name}_{self.epoch}.pkl"
        pickle.dump(self.model, open(path, "wb"))
        print(f"A model is saved successfully as {path}!")

    def load_model(self):
        dataset = "sample" if self.sample else "full"
        path = f"saved_models/{dataset}_{self.model_name}_{self.epoch}.pkl"

        try:
            self.model = pickle.load(open(path, "rb"))
            if torch.cuda.is_available():
                self.model.cuda()

            print(f"Model in {path} loaded successfully!")

            # return model
        except:
            print(f"No available model such as {path}.")
            exit()

