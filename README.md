## End-to-end Audio to NLU

# Notebooks

1. Data processing - This file contains the dataprocessing steps. We process the librispeech datasets to obtain the aligned MFCC features and the transcriptions for the seq2seq model.

2. Basic model - This file contain the model and training code for a simple end to end model. The encoder and decoders are both GRUs and we use a simple cross entopy loss for training.
