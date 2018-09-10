# ProGenDec_BLSTM
BSci project at Universidad de Murcia:

## Predicting genes related to intellectual disability applying Long Short Term Memory networks (LSTMs)

Program made for treating with gene/protein sequences, with the purpose to distinguish those genes that, when mutated, are the cause of a neurological disease. This problem is faced using LSTM networks, as they seem the appropriate choice for data that comes in the form of sequences.

The study carried out using ProGenDec_BLSTM used data from the England Genomics' PanelApp. Downloading the gene IDs from there, the R script get_data/get_data.R gets their correspondent nucleotids sequences. The resulting data are .fasta files, that the mains program reads and parse. As the path to the files are hardcoded in the scripts, you may want to change them to your local paths, both in the mentioned R script and in Parameters.py.

## Usage

ProGenDec_BLSTM.py is the python script to run, and its configuration parameters can either be modified in Parameters.py or in the command line, being these the available parameters:

-mode [int]: the data encoding to use. 0 correspond to simple sequences, 1 to window encoding, and 2 to one-hot encoding.

-genes True: to use gene sequences. It uses protein sequences by default.

-conv True: to use the hybrid model. It uses the BLSTM by default.

-seqLen [int]: sets the length of the sequences.

-window [int]: sets the size of the window (only useful for the window data preprocessing).

-lstm [int]: sets the number of cells of each LSTM layer.

-timedist [int]: sets the number of nodes of the time distributed dense layer.

-dropout [float]: dropout value to apply.

-recdropout [float]: recurrent dropout to apply in the LSTM layer.

-lr [float]: learning rate.

-batch [int]: size of the batches for training.

-epochs [int]: number of epochs to train on.

-test [run ID]: if present, the program will be in test only mode, so training won’t happen. It will look into the run folder 
with the name of the ID passed, and test the saved model with the best weights from the training process, using the test data.

# Credits
-Guillermo Castillo García
-Juan Antonio Botía Blaya
