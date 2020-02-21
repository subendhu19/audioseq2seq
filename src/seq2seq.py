import warnings

import random
import time
import multiprocessing as mp

import os
import gluonnlp as nlp
import pickle

import numpy as np
import mxnet as mx
from io import open
from mxnet import gluon, autograd
from mxnet.gluon import nn, rnn, Block
from mxnet import ndarray as F
from gluonnlp.model.transformer import TransformerEncoder
from gluonnlp.model.transformer import TransformerDecoder
from mxnet.gluon.loss import Loss as Loss
import argparse
import nltk

warnings.filterwarnings('ignore')
random.seed(123)
np.random.seed(123)
mx.random.seed(123)


def preprocess(x):
    name, audio, words = x
    split_words = ['<BOS>'] + words.split(' ') + ['<EOS>']
    return audio, np.array([vocabulary[word][0] for word in split_words]), float(len(audio)), float(len(split_words))


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


def get_dataloader(train_dataset, dev_dataset, test_dataset, train_data_lengths, batch_size, bucket_num, bucket_ratio):
    batchify_fn = nlp.data.batchify.Tuple(
        nlp.data.batchify.Pad(dtype='float32'),
        nlp.data.batchify.Pad(dtype='float32'),
        nlp.data.batchify.Stack(dtype='float32'),
        nlp.data.batchify.Stack(dtype='float32'))
    batch_sampler = nlp.data.sampler.FixedBucketSampler(
        train_data_lengths,
        batch_size=batch_size,
        num_buckets=bucket_num,
        ratio=bucket_ratio,
        shuffle=True)
    print(batch_sampler.stats())

    train_dataloader = gluon.data.DataLoader(
        dataset=train_dataset,
        batch_sampler=batch_sampler,
        batchify_fn=batchify_fn)
    dev_dataloader = gluon.data.DataLoader(
        dataset=dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        batchify_fn=batchify_fn)
    test_dataloader = gluon.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        batchify_fn=batchify_fn)
    return train_dataloader, dev_dataloader, test_dataloader


class TriangularSchedule:
    def __init__(self, min_lr, max_lr, cycle_length, inc_fraction=0.5):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.cycle_length = cycle_length
        self.inc_fraction = inc_fraction

    def __call__(self, iteration):
        if iteration <= self.cycle_length*self.inc_fraction:
            unit_cycle = iteration * 1 / (self.cycle_length * self.inc_fraction)
        elif iteration <= self.cycle_length:
            unit_cycle = (self.cycle_length - iteration) * 1 / (self.cycle_length * (1 - self.inc_fraction))
        else:
            unit_cycle = 0
        adjusted_cycle = (unit_cycle * (self.max_lr - self.min_lr)) + self.min_lr
        return adjusted_cycle


class SubSampler(gluon.HybridBlock):

    def __init__(self, size=3, prefix=None, params=None):
        super(SubSampler, self).__init__(prefix=prefix, params=params)
        self.size = size

    def forward(self, data, valid_length):
        masked_encoded = F.SequenceMask(data,
                                        sequence_length=valid_length,
                                        use_sequence_length=True)
        subsampled = F.Pooling(masked_encoded.swapaxes(0, 2), kernel=self.size, pool_type='max',
                               stride=self.size).swapaxes(0, 2)
        sub_valid_length = mx.nd.ceil(valid_length / self.size)
        return subsampled, sub_valid_length


class AudioEncoderRNN(Block):

    def __init__(self, input_size, hidden_size, context, sub_sample_size=3):
        super(AudioEncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sub_sample_size = sub_sample_size
        self.context = context

        with self.name_scope():
            self.proj = nn.Dense(hidden_size, flatten=False)
            self.rnn1 = rnn.GRU(hidden_size, input_size=self.hidden_size)
            self.subsampler = SubSampler(self.sub_sample_size)
            self.rnn2 = rnn.GRU(hidden_size, input_size=self.hidden_size)

    def forward(self, input, lengths):
        hidden = self.init_hidden(len(input), self.context)
        input = input.swapaxes(0, 1)
        input = self.proj(input)
        output, hidden1 = self.rnn1(input, hidden)
        subsampled, sub_lengths = self.subsampler(output, lengths)
        output, _ = self.rnn2(subsampled, hidden1)
        return output.swapaxes(0, 1), sub_lengths

    def init_hidden(self, batchsize, ctx):
        return [F.zeros((1, batchsize, self.hidden_size), ctx=ctx)]


class AudioEncoderTransformer(Block):

    def __init__(self, input_size, hidden_size, sub_sample_size=3):
        super(AudioEncoderTransformer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sub_sample_size = sub_sample_size

        with self.name_scope():
            self.proj = nn.Dense(hidden_size, flatten=False)
            self.t1 = TransformerEncoder(units=self.hidden_size, num_layers=8, hidden_size=self.hidden_size * 4,
                                         max_length=500,
                                         num_heads=8)
            self.subsampler = SubSampler(self.sub_sample_size)
            self.t2 = TransformerEncoder(units=self.hidden_size, num_layers=8, hidden_size=self.hidden_size * 4,
                                         max_length=500,
                                         num_heads=8)

    def forward(self, input, lengths):
        input = self.proj(input)
        output, _ = self.t1(input, None, lengths)
        output = output.swapaxes(0, 1)
        subsampled, sub_lengths = self.subsampler(output, lengths)
        subsampled = subsampled.swapaxes(0, 1)
        output, _ = self.t2(subsampled, None, sub_lengths)
        return output, sub_lengths


class AudioWordDecoder(Block):

    def __init__(self, output_size, hidden_size):
        super(AudioWordDecoder, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size

        with self.name_scope():
            self.embedding = nn.Embedding(output_size, hidden_size)
            self.t = TransformerDecoder(units=self.hidden_size, num_layers=1, hidden_size=self.hidden_size * 4,
                                        max_length=100,
                                        num_heads=1)
            self.out = nn.Dense(output_size, in_units=self.hidden_size, flatten=False)

    def forward(self, input, enc_outs, enc_valid_lengths, dec_valid_lengths):
        output = self.embedding(input)
        dec_states = self.t.init_state_from_encoder(enc_outs, enc_valid_lengths)
        output, _, _ = self.t.decode_seq(output, dec_states, dec_valid_lengths)
        output = self.out(output)
        return output


class Seq2Seq(Block):

    def __init__(self, input_size, output_size, enc_hidden_size, dec_hidden_size, context):
        super(Seq2Seq, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.context = context
        with self.name_scope():
            self.encoder = AudioEncoderRNN(input_size=input_size, hidden_size=enc_hidden_size, context=context)
            self.decoder = AudioWordDecoder(hidden_size=dec_hidden_size, output_size=output_size)

    def forward(self, audio, alengths, words, wlengths):
        encoder_outputs, encoder_out_lengths = self.encoder(audio, alengths)
        decoder_outputs = self.decoder(words, encoder_outputs, encoder_out_lengths, wlengths)
        return decoder_outputs


class SoftmaxSequenceCrossEntropyLoss(Loss):

    def __init__(self, axis=-1, sparse_label=True, from_logits=False, weight=None,
                 batch_axis=0, **kwargs):
        super(SoftmaxSequenceCrossEntropyLoss, self).__init__(
            weight, batch_axis, **kwargs)
        self._axis = axis
        self._sparse_label = sparse_label
        self._from_logits = from_logits

    def hybrid_forward(self, F, pred, label, valid_length):
        pred = pred[:, :-1, :]
        label = label[:, 1:]
        valid_length = valid_length - 1
        if not self._from_logits:
            pred = F.log_softmax(pred, self._axis)
        loss = mx.nd.squeeze(-F.pick(pred, label, axis=self._axis, keepdims=True), axis=2)
        loss = F.SequenceMask(loss.swapaxes(0, 1),
                              sequence_length=valid_length,
                              use_sequence_length=True).swapaxes(0, 1)
        return F.mean(loss, axis=self._batch_axis, exclude=True)


class BeamDecoder(object):

    def __init__(self, model):
        self._model = model

    def __call__(self, outputs, dec_states):
        outputs = self._model.decoder.embedding(outputs)
        outputs, new_states, _ = self._model.decoder.t(outputs, dec_states)
        return self._model.decoder.out(outputs), new_states


def get_bleu(s1, l1, s2, l2):
    s1 = mx.nd.cast(s1, dtype='int32')
    l1 = mx.nd.cast(l1, dtype='int32')
    s2 = mx.nd.cast(s2, dtype='int32')
    l2 = mx.nd.cast(l2, dtype='int32')

    scores = []
    for i in range(len(s1)):
        sent1 = [c.asscalar() for c in s1[i][:l1[i].asscalar()]]
        sent2 = [c.asscalar() for c in s2[i][:l2[i].asscalar()]]
        max_length = min(len(sent1), len(sent2))
        if max_length == 2:
            weights = [1.0 / 2] * 2
        elif max_length == 1:
            weights = [1]
        elif max_length == 3:
            weights = [1.0 / 3] * 3
        else:
            weights = [1.0 / 4] * 4
        scores.append(nltk.translate.bleu_score.sentence_bleu([sent1], sent2, weights))

    return np.mean(scores)


def get_sequence_accuracy(s1, l1, s2, l2, context):
    s1 = mx.nd.cast(s1, dtype='int32')
    l1 = mx.nd.cast(l1, dtype='int32')
    s2 = mx.nd.cast(s2, dtype='int32')
    l2 = mx.nd.cast(l2, dtype='int32')
    padding = mx.nd.zeros((s1.shape[0], abs(s1.shape[1] - s2.shape[1])), dtype=s2.dtype).as_in_context(context)
    if s1.shape[1] > s2.shape[1]:
        s2 = mx.nd.concat(s2, padding, dim=1)
    else:
        s1 = mx.nd.concat(s1, padding, dim=1)
    accs = F.SequenceMask((s1 == s2).swapaxes(0, 1),
                          sequence_length=mx.nd.minimum(l1, l2),
                          use_sequence_length=True)
    return (mx.nd.cast(accs.sum(), dtype='float32') / mx.nd.cast(l1.sum(), dtype='float32')).asnumpy().item()


def evaluate(net, context, test_dataloader, beam_sampler):
    for i, (audio, words, alength, wlength) in enumerate(test_dataloader):
        encoder_outputs, encoder_out_lengths = net.encoder(audio.as_in_context(context), alength.as_in_context(context))
        outputs = mx.nd.array([2] * words.shape[0]).as_in_context(context)
        decoder_states = net.decoder.t.init_state_from_encoder(encoder_outputs, encoder_out_lengths)
        samples, scores, valid_lengths = beam_sampler(outputs, decoder_states)
        best_samples = samples[:, 0, 1:]
        best_vlens = valid_lengths[:, 0]
        return get_bleu(words.as_in_context(context)[:, 1:], wlength.as_in_context(context) - 1,
                        best_samples, best_vlens - 1)


def train(net, context, epochs, learning_rate, log_interval, grad_clip, train_dataloader, test_dataloader,
          beam_sampler, checkpoint_dir):
    print('Starting Training...')

    # scheduler = TriangularSchedule(min_lr=learning_rate * 1e-2, max_lr=learning_rate, cycle_length=10,
    #                                inc_fraction=0.2)
    # optimizer = mx.optimizer.Adam(learning_rate=learning_rate, lr_scheduler=scheduler)
    optimizer = mx.optimizer.Adam(learning_rate=learning_rate)
    trainer = gluon.Trainer(net.collect_params(), optimizer=optimizer)
    loss = SoftmaxSequenceCrossEntropyLoss()

    parameters = net.collect_params().values()

    best_test_metrics = {'epoch': 0, 'metric': 0}

    for epoch in range(epochs):
        start_epoch_time = time.time()
        epoch_loss = 0.0
        epoch_sent_num = 0
        epoch_wc = 0

        start_log_interval_time = time.time()
        log_interval_wc = 0
        log_interval_sent_num = 0
        log_interval_loss = 0.0

        for i, (audio, words, alength, wlength) in enumerate(train_dataloader):
            wc = alength.sum().asscalar()
            log_interval_wc += wc
            epoch_wc += wc
            log_interval_sent_num += audio.shape[1]
            epoch_sent_num += audio.shape[1]
            with autograd.record():
                decoder_outputs = net(audio.as_in_context(context), alength.as_in_context(context),
                                      words.as_in_context(context), wlength.as_in_context(context))
                L = loss(decoder_outputs, words.as_in_context(context), wlength.as_in_context(context)).sum()
            L.backward()

            if grad_clip:
                gluon.utils.clip_global_norm(
                    [p.grad(context) for p in parameters],
                    grad_clip)

            trainer.step(1)
            log_interval_loss += L.asscalar()
            epoch_loss += L.asscalar()
            if (i + 1) % log_interval == 0:
                print(
                    '[Epoch {} Batch {}/{}] elapsed {:.2f} s, '
                    'avg loss {:.6f}, throughput {:.2f}K fps'.format(
                        epoch, i + 1, len(train_dataloader),
                        time.time() - start_log_interval_time,
                        log_interval_loss / log_interval_sent_num,
                        log_interval_wc / 1000 / (time.time() - start_log_interval_time)))
                start_log_interval_time = time.time()
                log_interval_wc = 0
                log_interval_sent_num = 0
                log_interval_loss = 0

        end_epoch_time = time.time()
        test_bleu = evaluate(net, context, test_dataloader, beam_sampler)
        print('[Epoch {}] train avg loss {:.6f}, test bleu {:.2f}, '
              'throughput {:.2f}K fps'.format(epoch, epoch_loss / epoch_sent_num, test_bleu,
                                              epoch_wc / 1000 / (end_epoch_time - start_epoch_time)))

        net.save_parameters(os.path.join(checkpoint_dir, 'epoch_{}.params'.format(epoch)))
        if test_bleu > best_test_metrics['metric']:
            best_test_metrics['epoch'] = epoch
            best_test_metrics['metric'] = test_bleu
            net.save_parameters(os.path.join(checkpoint_dir, 'best.params'))

    print('Training complete.')


def generate_sequences(sampler, inputs, begin_states, num_print_outcomes):
    samples, scores, valid_lengths = sampler(inputs, begin_states)
    print('Generation Result:')

    for sample_id in range(samples.shape[0]):
        sample = samples[sample_id].asnumpy()
        score = scores[sample_id].asnumpy()
        valid_length = valid_lengths[sample_id].asnumpy()

        for i in range(num_print_outcomes):
            sentence = []
            for ele in sample[i][:valid_length[i]]:
                sentence.append(vocabulary_inv[ele])
            print([' '.join(sentence), score[i]])


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", default="data", type=str, required=False,
                        help="The path to the data directory.")
    parser.add_argument("--checkpoint_dir", default="checkpoints", type=str, required=False,
                        help="The output directory where the model checkpoints will be written.")

    parser.add_argument("--batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--num_epochs", default=75, type=int,
                        help="Number of epochs for training.")
    parser.add_argument("--learning_rate", default=5e-3, type=float,
                        help="The initial learning rate.")

    args = parser.parse_args()

    # Data Processing
    train_dataset = pickle.load(open(os.path.join(args.data_dir, 'dev_processed.p'), 'rb'))[:10]
    dev_dataset = pickle.load(open(os.path.join(args.data_dir, 'dev_processed.p'), 'rb'))[:10]
    test_dataset = pickle.load(open(os.path.join(args.data_dir, 'dev_processed.p'), 'rb'))[:10]
    input_size = len(train_dataset[0][1][0])

    global vocabulary
    global vocabulary_inv

    if 'cached_vocab.p' in os.listdir(args.data_dir):
        vocabulary, vocabulary_inv = pickle.load(open(os.path.join(args.data_dir, 'cached_vocab.p'), 'rb'))
        print('Vocabulary loaded from cached file')
    else:
        vocabulary = {'<pad>': [0, 1], '<unk>': [1, 1], '<BOS>': [2, 1], '<EOS>': [3, 1]}
        for item in train_dataset + dev_dataset + test_dataset:
            words = item[2].split(' ')
            for word in words:
                if word in vocabulary:
                    vocabulary[word][1] += 1
                else:
                    vocabulary[word] = [len(vocabulary), 1]

        vocabulary_inv = {}
        for key in vocabulary:
            vocabulary_inv[vocabulary[key][0]] = key
        print('Vocabulary built for the first time')
        pickle.dump((vocabulary, vocabulary_inv), open(os.path.join(args.data_dir, 'cached_vocab.p'), 'wb'))

    vocab_list = [(word, vocabulary[word][1]) for word in vocabulary]
    print('Vocabulary Statistics:')
    print('Total: {}'.format(len(vocabulary)))

    vocab_list.sort(key=lambda x: x[1], reverse=True)
    print('Max count: {} ({})'.format(vocab_list[0][1], vocab_list[0][0]))
    print('Most frequent words: ')
    for i in range(5):
        print('\t{}\t({})'.format(vocab_list[i][0], vocab_list[i][1]))
    del vocab_list

    train_dataset, train_data_lengths = preprocess_dataset(train_dataset)
    dev_dataset, dev_data_lengths = preprocess_dataset(dev_dataset)
    test_dataset, test_data_lengths = preprocess_dataset(test_dataset)

    # Modeling
    learning_rate, batch_size = args.learning_rate, args.batch_size
    bucket_num, bucket_ratio = 10, 0.2
    grad_clip = None
    log_interval = 50
    epochs = args.num_epochs

    train_dataloader, dev_dataloader, test_dataloader = get_dataloader(train_dataset, dev_dataset, test_dataset,
                                                                       train_data_lengths,
                                                                       batch_size, bucket_num, bucket_ratio)
    # context = mx.gpu(0)
    context = mx.cpu()
    net = Seq2Seq(input_size=input_size, output_size=len(vocabulary), enc_hidden_size=16, dec_hidden_size=16,
                  context=context)
    net.initialize(mx.init.Xavier(), ctx=context)

    scorer = nlp.model.BeamSearchScorer(alpha=0, K=5, from_logits=False)
    eos_id = vocabulary['<EOS>'][0]
    beam_sampler = nlp.model.BeamSearchSampler(beam_size=5,
                                               decoder=BeamDecoder(net),
                                               eos_id=eos_id,
                                               scorer=scorer,
                                               max_length=50)

    train(net, context, epochs, learning_rate, log_interval, grad_clip, train_dataloader, dev_dataloader, beam_sampler,
          args.checkpoint_dir)
    net.save_parameters(os.path.join(args.checkpoint_dir, 'final.params'))

    beam_sampler = nlp.model.BeamSearchSampler(beam_size=5,
                                               decoder=BeamDecoder(net),
                                               eos_id=eos_id,
                                               scorer=scorer,
                                               max_length=100)

    net.load_parameters(filename=os.path.join(args.checkpoint_dir, 'best.params'), ctx=context)
    test_metric = evaluate(net, context, test_dataloader, beam_sampler)
    print('Test BLEU on best dev model: {}'.format(test_metric))

    print('Some decoder outputs: ')

    for i, (audio, words, alength, wlength) in enumerate(test_dataloader):
        encoder_outputs, encoder_out_lengths = net.encoder(audio.as_in_context(context), alength.as_in_context(context))
        outputs = mx.nd.array([2] * words.shape[0]).as_in_context(context)
        decoder_states = net.decoder.t.init_state_from_encoder(encoder_outputs, encoder_out_lengths)
        samples, scores, valid_lengths = beam_sampler(outputs, decoder_states)
        samples = samples[:, 0, 1:]
        valid_lengths = valid_lengths[:, 0] - 1

        for k in range(len(samples)):
            sample = words[k]
            slen = wlength[k].asscalar()

            sentence = []
            for ele in sample[:slen]:
                sentence.append(vocabulary_inv[ele.asscalar()])
            print('Gold:\t' + ' '.join(sentence[1:-1]))

            sample = samples[k]
            slen = valid_lengths[k].asscalar()

            sentence = []
            for ele in sample[:slen]:
                sentence.append(vocabulary_inv[ele.asscalar()])
            print('Pred:\t' + ' '.join(sentence[:-1]))


if __name__ == "__main__":
    main()
