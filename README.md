# End-to-end Audio to NLU

## Notebooks

1. Data preprocessing - This file contains the dataprocessing steps. We process the librispeech datasets to obtain the aligned MFCC features and the transcriptions for the seq2seq model. It also contains some plots and data analysis.

2. Seq2Seq model - This file contain the model and training code for a simple end to end model. The encoder and decoders are both GRUs and we use a simple cross entopy loss for training.
