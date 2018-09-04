import numpy as np
from Bio import SeqIO
from os import listdir
from os.path import isfile, join
import Parameters
from Bio.Seq import Seq
from Bio.Alphabet import generic_dna

class prepareData():
    
    def __init__(self):
        self.DATA_HOME = 'data/'
        self.window_size = Parameters.window_size
        self.sequenceLength = Parameters.sequence_length
        self.totalLength = 0
        self.numSeqs = 0
        self.codingPro = ['M', 'K', 'H', 'Y', 'E', 'Q', 'R', 'D', 'S', 'N', 'L', 'W', 'G', 'A', 'V', 'F', 'P', 'C', '*', 'T', 'I', 'X']
        self.codingGen = ['A', 'C', 'T', 'G', 'N']

        
    # Read and parse all the .fasta files in the path    
    def read_fasta(self, path):
        files = [f for f in listdir(path) if isfile(join(path, f))]
        coding_record = []
        names = []
        j = 0
        seqs = 0
        localLength = 0
        
        if Parameters.genes:
            coding = self.codingGen
        else:
            coding = self.codingPro
                
        for file in files:
            j += 1
            seq_record = list(SeqIO.parse(path+file, 'fasta'))
            
            for record in seq_record:
                finalSeq = []
                sequence = record.name
                splits = sequence.split('_')
                sequence = splits[3]
                lenSeq = 0
                if sequence[1] != 'e':
                    self.numSeqs += 1 
                    seqs += 1
                    names.append(splits[1])
                    
                    #If we want protein sequences, translate
                    if not Parameters.genes:                    
                        mRNA = Seq(sequence, generic_dna)
                        sequence = mRNA.translate()
                    # Store sequence length (for printing/analyzing)
                    lenSeq = len(sequence)
                    self.totalLength += lenSeq
                    localLength += lenSeq
                    # Set elements to be uppercase, and append to sequence
                    for aa in sequence:
                        aa = aa.upper()
                        if aa not in coding:
                            continue
                        finalSeq.append(coding.index(aa))
                    # Append the sequence to the list of sequences
                    if not finalSeq == "":
                        coding_record.append(finalSeq)
                    
        print('---------------------------------------------------------------------------')
        print('Directory:', path)
        print('Number of genes:', j)
        print('Readed sequences:', seqs)
        print('Average length =', localLength / seqs)
        print('Accumulated readed sequences:', self.numSeqs)
        print('Accumulated average length =', self.totalLength / self.numSeqs)
        return coding_record, names



    def genertateMat(self, path):   
        if Parameters.mode == 1:
            mats, protein_names = self.generateMat1(path)
        elif Parameters.mode == 2:
            mats, protein_names = self.generateMat2(path)
        else:
            mats, protein_names = self.generateMat0(path)
                
        print('Array shape =', mats.shape)
        print('---------------------------------------------------------------------------')
        
        return mats, protein_names
        
        
    
    # Simple
    def generateMat0(self, path):
        coding_record, protein_names  = self.read_fasta(path)     
        mats = np.asarray(np.full([len(coding_record), Parameters.sequence_length, 1], -1))
        index = 0
        for seq in coding_record:
            while len(seq) < Parameters.sequence_length:
                seq.append(-1)      # Fill with -1 if the sequence is shorter than max length
                
            seq = seq[:Parameters.sequence_length]  #Trime
            seq = np.expand_dims(seq, axis = 1)
            
            mats[index] = seq
            index = index + 1
        return mats, protein_names
        
        
        
    # Window
    def generateMat1(self, path):
        coding_record, protein_names  = self.read_fasta(path)
        wside = (self.window_size - 1)/2
        if Parameters.genes:
            lenCod = len(self.codingGen)
        else:
            lenCod = len(self.codingPro)
            
        mats = np.asarray(
            np.zeros([len(coding_record), self.sequenceLength, lenCod*self.window_size]))
        
        index = 0
        for seq in coding_record:
            seqLength = len(seq)
            if seqLength > self.sequenceLength:
                seqLength = self.sequenceLength
                
            inputVector= np.zeros([self.sequenceLength, lenCod*self.window_size])
            for i in range(self.sequenceLength):
                starti = i - wside  # Starting position of the window
                for j in range(self.window_size):
                    iter = starti + j
                    if (iter < 0 or iter >= seqLength): # If it is out of the sequene, next iteration
                        continue
                    aaCharPos = seq[int(iter)] # The element of the sequence
                    inputVector[i][j*lenCod+aaCharPos] += 1.0   # Adds one appearance of it in the corresponding position
                    
            mats[index] = inputVector
            index = index + 1
        return mats, protein_names
        
        
        
    # One-hot
    def generateMat2(self, path):   
        coding_record, protein_names  = self.read_fasta(path)
        
        if Parameters.genes:
            lenCod = len(self.codingGen)
        else:
            lenCod = len(self.codingPro)
            
        mats = np.asarray(
            np.zeros([len(coding_record), self.sequenceLength, lenCod]))
        index = 0
        for seq in coding_record:
            # One-hot encoding    
            seqMat = np.zeros([self.sequenceLength, lenCod])
            for i in range(len(seq)):
                if Parameters.genes and seq[i] in self.codingGen:
                    seqMat[i, self.codingGen.index(seq[i])] = 1.0
                elif seq[i] in self.codingPro:
                    seqMat[i, self.codingPro.index(seq[i])] = 1.0
            
            mats[index] = seqMat
            index = index + 1
        return mats, protein_names



    def generateInputData(self):
        if Parameters.genes:
            print('\n***********************************\n      GETTING GENE SEQUENCES\n***********************************\n')
        else:
            print('\n**************************************\n      GETTING PROTEIN SEQUENCES\n**************************************\n')
        pos_train_mat, pos_train_names = self.genertateMat(Parameters.pos_train_dir)
        neg_train_mat, neg_train_names = self.genertateMat(Parameters.neg_train_dir)
        pos_test_mat, pos_test_names = self.genertateMat(Parameters.pos_test_dir)
        neg_test_mat, neg_test_names = self.genertateMat(Parameters.neg_test_dir)
            
        print('Giving shape...')
        pos_train_num = pos_train_mat.shape[0]
        neg_train_num = neg_train_mat.shape[0]
        pos_test_num = pos_test_mat.shape[0]
        neg_test_num = neg_test_mat.shape[0]

        train_mat = np.vstack((pos_train_mat, neg_train_mat))
        test_mat = np.vstack((pos_test_mat, neg_test_mat))
        train_label = np.hstack((np.ones(pos_train_num), np.zeros(neg_train_num)))
        test_label = np.hstack((np.ones(pos_test_num), np.zeros(neg_test_num)))
        pos_test_names.extend(neg_test_names)
        test_names = pos_test_names
        
        train_num = pos_train_mat.shape[0] + neg_train_mat.shape[0]   
        test_num = pos_test_mat.shape[0] + neg_test_mat.shape[0]
        pos_train_percentage = pos_train_num / train_num
        neg_train_percentage = neg_train_num / train_num
        pos_test_percentage = pos_test_num / test_num
        neg_test_percentage = neg_test_num / test_num
        print('Done!')
        print('\nTrain:\tpos:', pos_train_percentage, '\tneg:', neg_train_percentage)
        print('Test:\tpos:', pos_test_percentage, '\tneg:', neg_test_percentage, '\n')
        
        return (train_mat, train_label, test_mat, test_label, test_names)



    def generateTestingSamples(self):
        if Parameters.genes:
            print('\n************************************\n   GETTING TESTING GENE SEQUENCES\n************************************\n')
        else:
            print('\n***************************************\n   GETTING TESTING PROTEIN SEQUENCES\n***************************************\n')
        pos_test_mat, pos_test_names = self.genertateMat(Parameters.pos_test_dir)
        neg_test_mat, neg_test_names = self.genertateMat(Parameters.neg_test_dir)

        print('Giving shape...')
        pos_test_num = pos_test_mat.shape[0]
        neg_test_num = neg_test_mat.shape[0]

        test_mat = np.vstack((pos_test_mat, neg_test_mat))
        test_label = np.hstack((np.ones(pos_test_num), np.zeros(neg_test_num)))
        pos_test_names.extend(neg_test_names)
        test_names = pos_test_names
        
        test_num = pos_test_mat.shape[0] + neg_test_mat.shape[0]
        pos_test_percentage = pos_test_num / test_num
        neg_test_percentage = neg_test_num / test_num
        
        print('Done!\n')
        print('Test:\tpos:', pos_test_percentage, '\tneg:', neg_test_percentage, '\n\n\n')
        
        return (test_mat, test_label, test_names)