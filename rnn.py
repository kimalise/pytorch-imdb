import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from imdb_dataset import load_imdb_data
from imdb_dataset import IMDBDataset


def create_variable(tensor):
    # Do cuda() before wrapping with variable
    if torch.cuda.is_available():
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)

class Rnn(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, output_size, n_layers=1, bidirectional=True):
        super(Rnn, self).__init__()
        self.n_layers = n_layers
        self.n_directions = int(bidirectional) + 1
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, n_layers, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, seq_lengths):

        batch_size = input.size(1)
        hidden = self._init_hidden(batch_size)

        embeded_seq_tensor = self.embedding(input)

        # seq_lengths = create_variable(seq_lengths)

        # pack them up nicely
        packed_input = pack_padded_sequence(
            embeded_seq_tensor, seq_lengths.data.cpu().numpy())

        packed_output, hidden = self.gru(packed_input, hidden)

        output = self.fc(hidden[0])

        # print("model output size: ", output.size())
        return output

    # 这一步不能忘掉，否则rnn在个数据进来时，一直会往后循环
    def _init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers * self.n_directions,
                             batch_size, self.hidden_size)
        return create_variable(hidden)


# def create_variable(tensor):
#     # Do cuda() before wrapping with variable
#     if torch.cuda.is_available():
#         return Variable(tensor.cuda())
#     else:
#         return Variable(tensor)

# dataset
data = load_imdb_data()
x_train, y_train, x_dev, y_dev, x_test, y_test, word2idx, idx2word = data

# parameters
batch_size = 32
vocab_size = len(word2idx)
hidden_size = 64
output_size = 2
n_epoch = 1

train_dataset = IMDBDataset(x_train[:], y_train[:], word2idx, idx2word, batch_size=batch_size, shuffle=True)

# model
classifier = Rnn(vocab_size=vocab_size, embedding_size=100, hidden_size=hidden_size, output_size=output_size)
if torch.cuda.is_available():
    classifier.cuda()

optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Train cycle
def train():
    total_loss = 0

    for epoch in range(1, n_epoch + 1):
        step = 1
        for seq_padded, seq_lengths, targets in train_dataset.batches():

            seq_padded = create_variable(torch.LongTensor(seq_padded))
            seq_lengths = create_variable(torch.LongTensor(seq_lengths))
            targets = create_variable(torch.LongTensor(targets))

            output = classifier(seq_padded, seq_lengths)

            # targets = create_variable(targets)
            loss = criterion(output, targets)
            total_loss += loss.data[0]

            classifier.zero_grad()
            loss.backward()
            optimizer.step()

            print("epoch ", epoch, "    step: ", step, "    loss: ", loss)

            step += 1

    return total_loss

if __name__ == '__main__':
    train()

