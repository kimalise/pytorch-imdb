
NUM_SAMPLE = 5000

imdb_neg_file = "./data/imdb.neg"
imdb_pos_file = "./data/imdb.pos"

sample_neg_file = "./data/imdb.sample.neg"
sample_pos_file = "./data/imdb.sample.pos"

def build_sample_dataset(origin_file, sample_file):
    with open(origin_file) as f:
        lines = f.readlines()

    with open(sample_file, "w") as f:
        f.write(''.join(lines[:NUM_SAMPLE]))


if __name__ == '__main__':
    build_sample_dataset(imdb_neg_file, sample_neg_file)
    build_sample_dataset(imdb_pos_file, sample_pos_file)
