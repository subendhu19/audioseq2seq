import soundfile as sf
from python_speech_features import logfbank, mfcc
import os
import pickle
import argparse
from tqdm import tqdm
import numpy as np


def compute_mfcc(audio_data, sample_rate, size):
    mfcc_feat = mfcc(audio_data, sample_rate, numcep=size, nfilt=size*2)
    return mfcc_feat - (np.mean(mfcc_feat, axis=0) + 1e-8)


def compute_lfb(audio_data, sample_rate, size):
    fbank_feat = logfbank(audio_data, sample_rate, nfilt=size)
    return fbank_feat - (np.mean(fbank_feat, axis=0) + 1e-8)


def get_mfcc(audio_file, size=13):
    audio, sample_rate = sf.read(audio_file)
    feats = compute_mfcc(audio, sample_rate, size)
    return feats


def get_lfb(audio_file, size=26):
    audio, sample_rate = sf.read(audio_file)
    feats = compute_lfb(audio, sample_rate, size)
    return feats


def process(path, split, feature_type='lfb', size=26):
    root_folder = os.path.join(path, split)
    processed_set = []
    all_files = []

    for folder in os.listdir(root_folder):
        if '.' not in folder:
            for subfolder in os.listdir(os.path.join(root_folder, folder)):
                if '.' not in subfolder:
                    txtfilename = folder + '-' + subfolder + '.trans.bpe.txt'
                    txtfile = os.path.join(root_folder, folder, subfolder, txtfilename)
                    all_files.append((root_folder, folder, subfolder, txtfile))
    
    for (root_folder, folder, subfolder, txtfile) in tqdm(all_files):                
        with open(txtfile) as efile:
            for line in efile:
                words = line.strip().split(' ')
                transcription = ' '.join(words[1:])
                audio_file = words[0] + '.flac'
                if feature_type == 'lfb':
                    audio_features = get_lfb(os.path.join(root_folder, folder, subfolder, audio_file), size=size)
                else:
                    audio_features = get_mfcc(os.path.join(root_folder, folder, subfolder, audio_file), size=size)
                processed_set.append((os.path.join(root_folder, folder, subfolder, audio_file), audio_features,
                                      transcription))

    pickle.dump(processed_set, open(os.path.join(path, '{}.{}.{}.p'.format(split, feature_type, size)), 'wb'))
    print('Processed {} set: {} instances'.format(split, len(processed_set)))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--path", default=None, type=str, required=True,
                        help="The path to the data folder.")
    parser.add_argument("--feature_type", default='lfb', type=str,
                        help="Type of features for the audio: LFB or MFCC.")
    parser.add_argument("--size", default=26, type=int,
                        help="Size of the audio features.")
    args = parser.parse_args()

    splits = ['dev-clean', 'test-clean', 'train-clean-100', 'train-clean-360']
    for split in splits:
        process(args.path, split, args.feature_type, args.size)


if __name__ == "__main__":
    main()
