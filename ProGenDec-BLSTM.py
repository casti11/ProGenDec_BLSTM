import os
import argparse
import inspect
import Parameters
import PrepareData
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, LSTM, Conv1D, MaxPooling1D
from keras.layers.wrappers import TimeDistributed, Bidirectional
from os import listdir
from keras.models import model_from_json
from keras import optimizers
from time import gmtime, strftime
from sklearn.metrics import roc_auc_score
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

rocValues = []
bestAcc = 0



# Define the model
def get_model():
    sequence_length = Parameters.sequence_length
    latent_dim = Parameters.lstm_cells
    window_size = Parameters.window_size
    # Genes or proteins
    if Parameters.genes:
        alphabetLength = 5
    else:
        alphabetLength = 22
    # Type of data encoding -> input shape
    if Parameters.mode == 1:
        inshape = (sequence_length, alphabetLength * window_size)
    elif Parameters.mode == 2:
        inshape = (sequence_length, alphabetLength)
    else:
        inshape = (sequence_length, 1)
        
    # MODEL
    model = Sequential()

    if Parameters.convModel: # HYBRID MODEL
        model.add(Conv1D(filters = 80, kernel_size = 20, input_shape=inshape))
        model.add(MaxPooling1D(pool_size = 10))
        model.add(Dropout(0.2))
        model.add(Bidirectional(LSTM(latent_dim, dropout=Parameters.dropout, recurrent_dropout=Parameters.recurrent_dropout, return_sequences=True), 
                                merge_mode='concat', weights=None),)
        model.add(TimeDistributed(Dense(Parameters.timeDist_nodes, activation='relu')))
        model.add(Flatten())
    else:   # BLSTM MODEL
        model.add(Bidirectional(LSTM(latent_dim, dropout=Parameters.dropout, recurrent_dropout=Parameters.recurrent_dropout, return_sequences=True), 
                                merge_mode='concat', weights=None,input_shape=inshape),)
        model.add(TimeDistributed(Dense(Parameters.timeDist_nodes, activation='relu')))
        model.add(Flatten())
        
    # Common output layer
    model.add(Dense(1, activation='sigmoid'))
    
    optimizer = optimizers.rmsprop(lr=Parameters.learning_rate)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model



# Writes the results to a .txt file: accuracy, roc, individual predictions made
def writeResults(test_names, test_label, best_score, best_predicted_label, target_fam, best_roc, bestAcc, test):
    if test:
        writer = open(Parameters.results_path + 'result_test.txt', 'w')
    else:
        writer = open(Parameters.results_path + 'result.txt', 'w')
    writer.write('Test accuracy = ' + str(bestAcc) + '\n')
    writer.write('roc:\t' + str(best_roc) + '\n')
    writer.write('\nprotein\ttrue label\tscore\tprediction\n')
    
    tp, tn, fp, fn = 0.0, 0.0, 0.0, 0.0
    for j in range(len(test_names)):
        writer.write(test_names[j] + '\t' + str(test_label[j]) + '\t' + str(best_score[j][0]) + '\t' + str(
            best_predicted_label[j][0]) + '\n')
        if test_label[j] == 0:
            if best_predicted_label[j][0] == 0:
                tn += 1.0
            else:
                fp += 1.0
        else:
            if best_predicted_label[j][0] == 0:
                fn += 1.0
            else:
                tp += 1.0
                
    writer.write('\n\n\nTrue positives: ' + str(tp) + '\t' + str(tp/len(test_names)*100) + '%' +
        '\nFalse positives: ' + str(fp) + '\t' + str(fp/len(test_names)*100) + '%' +
        '\nTrue negatives: ' + str(tn) + '\t' + str(tn/len(test_names)*100) + '%' +
        '\nFalse negatives: ' + str(fn) + '\t' + str(fn/len(test_names)*100) + '%')
            
            

# Writes the parameters used in a .txt file in the directory corresponding to the run
def writeParamsModel():
    writer = open(Parameters.run_path + 'params & model.txt', 'w')
    writer.write('\n\n---------------------------------------- PARAMETERS ---------------------------------------\n' + 'Mode = ' + str(Parameters.mode) +
            '\nSequence length = ' + str(Parameters.sequence_length) + '\nEpochs = ' + str(Parameters.nb_of_epochs) + '\nLSTM cells = ' +
            str(Parameters.lstm_cells) + '\nTimeDistributed nodes = ' + str(Parameters.timeDist_nodes) + 'Dropout = ' + str(Parameters.dropout) +
            '\nRecurrent dropout = ' + str(Parameters.recurrent_dropout) + '\nGenes = ' + str(Parameters.genes) + '\nConvolutional model = ' +
            str(Parameters.convModel) + '\n---------------------------------------------------------------------------\n')
    writer.write('\n\n---------------------------------------- MODEL ---------------------------------------\n' + inspect.getsource(get_model) +
                 '\n---------------------------------------------------------------------------\n')



# Train the model
def TrainingModel():
    nb_epochs = Parameters.nb_of_epochs
    prepare = PrepareData.prepareData()    
    train, train_lable, test, test_label, test_names = prepare.generateInputData()
    best_roc = -1
    
    # Construct the neural network
    model = get_model()
    print(model.summary())
    
    # Save the structure of the neural network
    model_json = model.to_json()
    with open(Parameters.model_path + 'model' + ".json", "w") as json_file:
        json_file.write(model_json)
        
    writeParamsModel()
    train_acc = []
    train_loss = []
    test_acc = []
    test_loss = []
    
    # Train
    for epoch in range(nb_epochs):
        print ('\nEPOCH:', epoch+1, 'of', nb_epochs)
        history = model.fit(train, train_lable, epochs=1, shuffle=True, batch_size = Parameters.batch_size)
        loss, acc = model.evaluate(test, test_label)
        # Save training accuracy and loss
        train_acc.append(history.history.get('acc')[0])
        train_loss.append(history.history.get('loss')[0])
        # Print and save test acc and loss
        print('\tTest - loss: ', loss, '- acc:', acc)
        test_acc.append(acc)
        test_loss.append(loss)
        # Predict
        score = model.predict(test)
        predicted_label = model.predict_classes(test)
        unique, counts = np.unique(predicted_label, return_counts=True)
        # Print number of 0s and 1s predicted
        print("\tPredicted labels:", dict(zip(unique, counts)))
        # Roc
        roc = roc_auc_score(test_label, score)
        rocValues.append(roc)
        print ('\tROC =', roc)  
        
        # Select the epoch with the best performance
        if roc >= best_roc:
            best_roc = roc
            save_epoch = epoch
            best_score = score
            model.save_weights(Parameters.weights_path + str(save_epoch) + ".h5")
            best_predicted_label = predicted_label
            bestAcc = acc
        # Stop is too much overfitting appears
        save_plots(train_acc, train_loss, test_acc, test_loss, rocValues)
        if history.history.get('acc')[0] - acc > 0.2:
            break

    print ('Run:', Parameters.run)
    print ('Performance of ProDec-BLSTM: roc:', best_roc)

    writeResults(test_names, test_label, best_score, best_predicted_label, Parameters.run, best_roc, bestAcc, False)
    return train_acc, train_loss, test_acc, test_loss



# Test only, no training
def performPredictions():
    model_path = Parameters.model_path + 'model.json'
    weights_path = Parameters.weights_path
    prepare = PrepareData.prepareData()
    test, test_label, test_names = prepare.generateTestingSamples()

    # load model file
    jason_file = open(model_path, 'r')
    loaded_jason_file = jason_file.read()
    jason_file.close()
    model = model_from_json(loaded_jason_file)

    # load weights file
    lastWeights = 0
    for f in listdir(weights_path):
        epoch = (weights_path + f).split('/')[-1].split('.')[0]
        if int(epoch) > lastWeights:
            lastWeights = int(epoch)    
    model.load_weights(weights_path + str(lastWeights) + '.h5')
    
    optimizer = optimizers.rmsprop(lr=Parameters.learning_rate)    
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # perform predictions
    loss, acc = model.evaluate(test, test_label)
    score = model.predict(test)
    # Print number of 0s and 1s predicted
    predicted_label = model.predict_classes(test)
    unique, counts = np.unique(predicted_label, return_counts=True)
    print("\tPredicted labels:", dict(zip(unique, counts)))
    # ROC
    roc = roc_auc_score(test_label, score)

    print ('Run:', Parameters.run)
    print ('Performance of ProDec-BLSTM: roc:', roc)
    # save predictions to disk
    writeResults(test_names, test_label, score, predicted_label, Parameters.run, roc, acc, True)



def save_plots(trainAcc, trainLoss, testAcc, testLoss, auc):
    if os.path.isfile(Parameters.results_path + 'acc.png'):
        os.remove(Parameters.results_path + 'acc.png')
    if os.path.isfile(Parameters.results_path + 'loss.png'):
        os.remove(Parameters.results_path + 'loss.png')
    if os.path.isfile(Parameters.results_path + 'roc.png'):
        os.remove(Parameters.results_path + 'roc.png')
        
    plt.plot(list(range(len(trainAcc))), trainAcc, label = 'Training acc')
    plt.plot(list(range(len(trainAcc))), testAcc, label = 'Test acc')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='best')
    plt.savefig(Parameters.results_path + 'acc.png')
    plt.clf()    
    plt.plot(list(range(len(trainAcc))), trainLoss, label = 'Training loss')
    plt.plot(list(range(len(trainAcc))), testLoss, label = 'Test loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='best')
    plt.savefig(Parameters.results_path + 'loss.png')
    plt.clf()    
    plt.plot(list(range(len(trainAcc))), auc, label = 'ROC (AUC)')
    plt.ylabel('Area Under the Curve')
    plt.xlabel('Epoch')
    plt.legend(loc='best')
    plt.savefig(Parameters.results_path + 'roc.png')
    plt.clf()



def makeDirs():
    if not os.path.exists(Parameters.base_path):
        os.mkdir(Parameters.base_path)
    if not os.path.exists(Parameters.run_path):
        os.mkdir(Parameters.run_path)
    if not os.path.exists(Parameters.results_path):
        os.mkdir(Parameters.results_path)
    if not os.path.exists(Parameters.model_path):
        os.mkdir(Parameters.model_path)
    if not os.path.exists(Parameters.weights_path):
        os.mkdir(Parameters.weights_path)


        
def updatePaths():
    if Parameters.train:
        Parameters.run = strftime(Parameters.modeStr + "%Y-%m-%d %H:%M:%S", gmtime())
    Parameters.run_path = Parameters.base_path + Parameters.run + '/'
    Parameters.results_path = Parameters.run_path + 'result/'
    Parameters.model_path = Parameters.run_path + 'model/'
    Parameters.weights_path = Parameters.run_path + 'weights/'
        
        

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', type=int, help='Use simple sequences (0), window (1) or one-hot encoding (2)')
    parser.add_argument('-genes', type=bool, help='Use genes (true) or proteins (false)')
    parser.add_argument('-conv', type=bool, help='Use convolutional layer (true) or not (false)')
    parser.add_argument('-seqlen', type=int, help='Length of the sequences')
    parser.add_argument('-window', type=int, help='Length of the window')
    parser.add_argument('-lstm', type=int, help='Number of cells of the LSTM layer')
    parser.add_argument('-timedist', type=int, help='Number of nodes of the TimeDistributed dense layer')
    parser.add_argument('-dropout', type=float, help='Dropout value')
    parser.add_argument('-recdropout', type=float, help='Recurrent dropout value')
    parser.add_argument('-lr', type=float, help='Learning rate')
    parser.add_argument('-batch', type=int, help='Batch size')
    parser.add_argument('-epochs', type=int, help='Number of epochs')
    parser.add_argument('-test', type=str, help='Path to the run to test')
    args = parser.parse_args()
    if args.mode != None:
        Parameters.mode = args.mode
        if args.mode == 1:
            Parameters.modeStr = 'w_'
            updatePaths()
        elif args.mode == 2:
            Parameters.modeStr = 'o_'
            updatePaths()
    if args.genes != None:
        Parameters.genes = args.genes
    if args.conv != None:
        Parameters.convModel = args.conv
    if args.seqlen != None:
        Parameters.sequence_length = args.seqlen
    if args.window != None:
        Parameters.window_size = args.window
    if args.lstm != None:
        Parameters.lstm_cells = args.lstm
    if args.timedist != None:
        Parameters.timeDist_nodes = args.timedist
    if args.dropout != None:
        Parameters.dropout = args.dropout
    if args.recdropout != None:
        Parameters.recurrent_dropout = args.recdropout
    if args.lr != None:
        Parameters.learning_rate = args.lr
    if args.batch != None:
        Parameters.batch_size = args.batch
    if args.epochs != None:
        Parameters.nb_of_epochs = args.epochs
    if args.test != None:
        Parameters.test = True
        Parameters.train = False
        Parameters.run = args.test
        updatePaths()
    
    


if __name__ == '__main__':
    parseArguments()    
    
    if Parameters.convModel:
        device = '/gpu:0'
    else:
        device = '/cpu:0'
        
    with tf.device(device):
        if Parameters.train == True:
            makeDirs()
            TrainingModel()
        elif Parameters.test == True:
            performPredictions()
