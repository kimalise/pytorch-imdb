import argparse
from model_runner import ModelRunner
'''
scripts:
python main.py --model simple-gru --mode train --epoch 10 --batch_size 64

'''

def main():
    parser = argparse.ArgumentParser(description="-----[IMDB-classifier]-----")
    parser.add_argument("--sample", default=False, action='store_true', help="flag whether use sample dataset")
    parser.add_argument("--mode", default="train", help="train: train (with test) a model / test: test saved models")
    parser.add_argument("--model", default="simple-gru", help="available models: simple-gru, ...")
    parser.add_argument("--epoch", default=10, type=int, help="number of max epoch")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="learning rate")
    parser.add_argument("--batch_size", default=32, type=int, help="batch size")

    options = parser.parse_args()

    params = {
        'sample' : options.sample,
        'model': options.model,
        'mode' : options.mode,
        'batch_size' : options.batch_size,
        'epoch' : options.epoch,
        'learning_rate' : options.learning_rate
    }

    modelRunner = ModelRunner(params)
    if options.mode == 'train':
        print("=" * 20 + "TRAINING STARTED" + "=" * 20)
        modelRunner.train()
    elif options.mode == 'test':
        print("=" * 20 + "TESTING STARTED" + "=" * 20)
        modelRunner.load_model()
        modelRunner.test()

if __name__ == '__main__':
    main()


