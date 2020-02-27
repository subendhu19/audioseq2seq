import argparse
import pickle
import os
import time
import multiprocessing as mp
import mxnet as mx
from mxnet import gluon
import numpy as np
import gluonnlp as nlp
from src.train_and_test import Seq2Seq, BeamDecoder, evaluate


def preprocess(x):
    name, audio, words = x
    split_words = ['<BOS>'] + words.split(' ') + ['<EOS>']
    return -1 * audio, np.array([vocabulary[word][0] for word in split_words]), float(len(audio)), float(len(split_words))


def get_length(x):
    return float(len(x[1]))


def preprocess_dataset(dataset):
    start = time.time()
    with mp.Pool(5) as pool:
        dataset = gluon.data.SimpleDataset(pool.map(preprocess, dataset))
        lengths = gluon.data.SimpleDataset(pool.map(get_length, dataset))
    end = time.time()
    print('Done! Processing Time={:.2f}s, #Samples={}'.format(end - start, len(dataset)))
    return dataset, lengths


def get_dataloader(test_dataset, test_data_lengths, batch_size, bucket_num, bucket_ratio,
                   batch_size_per_gpu):
    batchify_fn = nlp.data.batchify.Tuple(
        nlp.data.batchify.Pad(dtype='float32'),
        nlp.data.batchify.Pad(dtype='float32'),
        nlp.data.batchify.Stack(dtype='float32'),
        nlp.data.batchify.Stack(dtype='float32'))
    batch_sampler = nlp.data.sampler.FixedBucketSampler(
        test_data_lengths,
        batch_size=batch_size,
        num_buckets=bucket_num,
        ratio=bucket_ratio,
        shuffle=True)
    print(batch_sampler.stats())

    test_dataloader = gluon.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size_per_gpu,
        shuffle=False,
        batchify_fn=batchify_fn)
    return test_dataloader


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", default="data", type=str, required=False,
                        help="The path to the data directory.")
    parser.add_argument("--checkpoint_dir", default="checkpoints", type=str, required=False,
                        help="The output directory where the model checkpoints will be written.")

    parser.add_argument("--batch_size_per_gpu", default=12, type=int,
                        help="Batch size per GPU/CPU for training.")

    parser.add_argument("--outfile", default="test_predictions.txt", type=str, required=False,
                        help="The name of the output file.")

    args = parser.parse_args()

    args.batch_size = args.batch_size_per_gpu * max(1, args.gpu_count)

    # Data Processing
    test_dataset = pickle.load(open(os.path.join(args.data_dir, 'test-clean-80.p'), 'rb'))

    input_size = len(test_dataset[0][1][0])

    global vocabulary
    global vocabulary_inv

    vocabulary, vocabulary_inv = pickle.load(open(os.path.join(args.data_dir, 'cached_vocab.p'), 'rb'))
    print('Vocabulary loaded from cached file')

    test_dataset, test_data_lengths = preprocess_dataset(test_dataset)

    # Modeling
    learning_rate, batch_size, batch_size_per_gpu = args.learning_rate, args.batch_size, args.batch_size_per_gpu
    bucket_num, bucket_ratio = 10, 0.2

    train_dataloader, dev_dataloader, test_dataloader = get_dataloader(test_dataset,
                                                                       test_data_lengths,
                                                                       batch_size, bucket_num, bucket_ratio,
                                                                       batch_size_per_gpu)
    if args.gpu_count == 0:
        context = mx.cpu()
    else:
        context = [mx.gpu(i) for i in range(args.gpu_count)]

    print('\nRunning on {}\n'.format(context))

    net = Seq2Seq(input_size=input_size, output_size=len(vocabulary), enc_hidden_size=1024, dec_hidden_size=1024)
    net.initialize(mx.init.Xavier(), ctx=context)

    scorer = nlp.model.BeamSearchScorer(alpha=0, K=5, from_logits=False)
    eos_id = vocabulary['<EOS>'][0]

    beam_sampler = nlp.model.BeamSearchSampler(beam_size=5,
                                               decoder=BeamDecoder(net),
                                               eos_id=eos_id,
                                               scorer=scorer,
                                               max_length=50)

    net.load_parameters(filename=os.path.join(args.checkpoint_dir, 'best.params'), ctx=context)

    print('Writing test output to file...')
    with open(args.outfile, 'w') as test_out:
        test_metric = evaluate(net, context, test_dataloader, beam_sampler)
        test_out.write('Test WER on best dev model: {}'.format(test_metric) + '\n\n')

        test_out.write('Predictions: \n\n')
        if context != mx.cpu():
            context = mx.gpu(0)
        for i, (audio, words, alength, wlength) in enumerate(test_dataloader):
            encoder_outputs, encoder_out_lengths = net.encoder(audio.as_in_context(context).expand_dims(1),
                                                               alength.as_in_context(context))
            outputs = mx.nd.array([2] * words.shape[0]).as_in_context(context)
            decoder_states = net.decoder.transformer.init_state_from_encoder(encoder_outputs, encoder_out_lengths)
            samples, scores, valid_lengths = beam_sampler(outputs, decoder_states)
            samples = samples[:, 0, 1:]
            valid_lengths = valid_lengths[:, 0] - 1

            for k in range(len(samples)):
                sample = words[k]
                slen = wlength[k].asscalar()

                sentence = []
                for ele in sample[:slen]:
                    sentence.append(vocabulary_inv[ele.asscalar()])
                test_out.write('Gold:\t' + ' '.join(sentence[1:-1]) + '\n')

                sample = samples[k]
                slen = valid_lengths[k].asscalar()

                sentence = []
                for ele in sample[:slen]:
                    sentence.append(vocabulary_inv[ele.asscalar()])
                test_out.write('Pred:\t' + ' '.join(sentence[:-1]) + '\n\n')
    print('All done')


if __name__ == "__main__":
    main()
