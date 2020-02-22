import soundfile as sf
from python_speech_features import logfbank
import os
import pickle
import argparse
from tqdm import tqdm


def compute_lfb(audio_data, sample_rate):
    fbank_feat = logfbank(audio_data, sample_rate, nfilt=80)
    return fbank_feat


def get_lfb(audio_file):
    audio, sample_rate = sf.read(audio_file)
    feats = compute_lfb(audio, sample_rate)
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
                audio_features = get_lfb(os.path.join(root_folder, folder, subfolder, audio_file))
                processed_set.append((os.path.join(root_folder, folder, subfolder, audio_file), audio_features,
                                      transcription))

    pickle.dump(processed_set, open(os.path.join(path, split+'-13.p'), 'wb'))
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
