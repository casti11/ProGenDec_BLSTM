from time import gmtime, strftime

mode = 0
sequence_length = 500
window_size = 3
nb_of_epochs = 150

lstm_cells = 128
timeDist_nodes = 16

batch_size = 100
learning_rate = 0.001
dropout = 0.5
recurrent_dropout = 0.5

genes = False
convModel = False

modeStr = 's_'
run = strftime(modeStr + "%Y-%m-%d_%H:%M:%S", gmtime())

train = True
test = False

base_path = 'runs/'
run_path = base_path + run + '/'
results_path = run_path + 'result/'
model_path = run_path + 'model/'
weights_path = run_path + 'weights/'

pos_train_dir = 'data/pos_train/'
neg_train_dir = 'data/neg_train/'
pos_test_dir = 'data/pos_test/'
neg_test_dir = 'data/neg_test/'
