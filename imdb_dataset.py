import json
import numpy as np
# import torch
# from torch.autograd import Variable

SEED = 1

PAD = '<PAD>'
OOV = '<OOV>'

def read_file(file_name):
    lines = []
    with open(file_name) as f:
        lines = f.readlines()
    return lines

def build_vocab(sents):
    word2idx = { PAD : 0, OOV : 1 }
    idx2word = { 0 : PAD, 1 : OOV }
    for sent in sents:
        for token in sent.split(' '):
            if token not in word2idx:
                new_idx = len(word2idx)
                word2idx[token] = new_idx
                idx2word[new_idx] = token

    return word2idx, idx2word

def convert_sentences_to_index(sents, word2idx):
    return [[word2idx[token] for token in sent.split(' ')] for sent in sents]
    # sents_idx = []
    # for sent in sents:
    #     sents_idx.append([word2idx[token] for token in sent.split(' ')])
    #
    # return sents_idx

def load_imdb_data(sample=False):
    print('start loading imdb data')

    if sample:
        imdb_neg_file = "./data/imdb.sample.neg"
        imdb_pos_file = "./data/imdb.sample.pos"
        # train 9000 + dev 1000, test 2000
        DEV_INDEX = 9000
    else:
        imdb_neg_file = "./data/imdb.neg"
        imdb_pos_file = "./data/imdb.pos"
        # train 550000 + dev 50000, test 2000
        DEV_INDEX = 550000

    neg_sents = read_file(imdb_neg_file)
    pos_sents = read_file(imdb_pos_file)

    sents = neg_sents + pos_sents
    lables = [0] * len(neg_sents) + [1] * len(pos_sents)

    np.random.seed(SEED)
    indices = np.arange(len(sents))
    np.random.shuffle(indices)

    sents = np.array(sents)[indices]
    labels = np.array(lables)[indices]

    # build vocabulary
    word2idx, idx2word = build_vocab(sents)

    # convert sentences to index
    sents_idx = convert_sentences_to_index(sents, word2idx)

    x_train, y_train = sents_idx[:DEV_INDEX], labels[:DEV_INDEX]
    x_dev, y_dev = sents_idx[DEV_INDEX:], labels[DEV_INDEX:]

    # load test data
    x_test, y_test = load_imdb_test_data(word2idx)

    print('end of loading imdb data')
    return x_train, y_train, x_dev, y_dev, x_test, y_test, word2idx, idx2word

def load_imdb_test_data(word2idx):
    imdb_test_file = "./data/rt_critics.test"
    with open(imdb_test_file) as f:
        lines = f.readlines()
        sents = []
        labels = []
        line_idx = -1
        # cur_idx = 0
        for line in lines:
            line_idx += 1
            if line_idx % 3 == 0:
                sent_idx = [word2idx[token] if token in word2idx else word2idx[OOV] for token in line.split(' ')]
                sents.append(sent_idx)
            elif line_idx % 3 == 1:
                labels.append( (1 if line == '+1' else 0) )
                # cur_idx += 1
            if line.strip() == '':
                continue
    return sents, labels


class IMDBDataset:
    def __init__(self, inputs, outputs, word2idx, idx2word, batch_size=32, shuffle=True):
        self.inputs = inputs
        self.outputs = outputs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.input_lens = [len(s) for s in inputs]
        self.word2idx = word2idx
        self.idx2word = idx2word

    def batches(self):
        assert len(self.inputs) == len(self.outputs)

        if self.shuffle:
            indices = np.arange(len(self.inputs), dtype=np.int32)
            np.random.shuffle(indices)

        for start_idx in range(0, len(self.inputs), self.batch_size):
            end_idx = min(start_idx + self.batch_size, len(self.inputs))
            if self.shuffle:
                excerpt = indices[start_idx : start_idx + self.batch_size]
                # excerpt = excerpt.tolist()
            else:
                excerpt = slice(start_idx, end_idx)

            seq_padded, seq_lengths, outputs = self._make_padding(np.array(self.inputs)[excerpt], np.array(self.input_lens)[excerpt], np.array(self.outputs)[excerpt])
            yield seq_padded, seq_lengths, outputs

    def _make_padding(self, vectorized_seqs, seq_lengths, outputs):

        seq_padded = np.zeros((len(vectorized_seqs), max(seq_lengths)), dtype=np.int32)

        for idx, (seq, seqlen) in enumerate(zip(vectorized_seqs, seq_lengths)):
            seq_padded[idx, :seqlen] = seq

        # print("seq_tensor", seq_tensor)

        perm_idx = np.argsort(seq_lengths)
        perm_idx = perm_idx[::-1] # reverse

        seq_lengths = seq_lengths[perm_idx]
        seq_padded = seq_padded[perm_idx]
        outputs = outputs[perm_idx]

        # seq_tensor = seq_tensor.transpose(0, 1)  # (B,L,D) -> (L,B,D)
        seq_padded = np.transpose(seq_padded)

        return seq_padded, seq_lengths, outputs

if __name__ == '__main__':

    data = load_imdb_data(sample=True)
    x_train, y_train, x_dev, y_dev, x_test, y_test, word2idx, idx2word = data

    train_dataset = IMDBDataset(x_train[:15], y_train[:15], word2idx, idx2word, batch_size=10, shuffle=True)
    # for batch_x, batch_y, batch_x_lens in train_dataset.batches():
    #     print(batch_x)
    #     print(batch_y)
    #     print(batch_x_lens)

    for seq_tensor, seq_lengths, targets in train_dataset.batches():
        print(seq_tensor)
        print(seq_lengths)
        print(targets)


