import soundfile as sf
import numpy as np
from python_speech_features import mfcc
import os
import pickle
import argparse
from tqdm import tqdm


def compute_mfcc(audio_data, sample_rate):
    audio_data = audio_data - np.mean(audio_data)
    audio_data = audio_data / np.max(audio_data)
    mfcc_feat = mfcc(audio_data, sample_rate, winlen=0.025, winstep=0.01,
                     numcep=128, nfilt=256, nfft=512, lowfreq=0, highfreq=None,
                     preemph=0.97, ceplifter=22, appendEnergy=True)
    return mfcc_feat 


def get_mfcc(audio_file):
    audio, sample_rate = sf.read(audio_file)
    feats = compute_mfcc(audio, sample_rate)
    return feats


def process(path, split):
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
                audio_features = get_mfcc(os.path.join(root_folder, folder, subfolder, audio_file))
                processed_set.append((os.path.join(root_folder, folder, subfolder, audio_file), audio_features,
                                      transcription))

    pickle.dump(processed_set, open(os.path.join(path, split+'.p'), 'wb'))
    print('Processed {} set: {} instances'.format(split, len(processed_set)))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--path", default=None, type=str, required=True,
                        help="The path to the data folder.")
    args = parser.parse_args()

    splits = ['dev-clean', 'test-clean', 'train-clean-100', 'train-clean-360']
    for split in splits:
        process(args.path, split)


if __name__ == "__main__":
    main()
