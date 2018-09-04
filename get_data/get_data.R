#!/usr/bin/env Rscript
library("biomaRt")
library(seqinr)
library(abind)
library(rjson)

dir.create("data/pos_seqs/")
dir.create("data/neg_seqs/")

# Positive genes
pos_table <- read.csv(file = "data/Intellectual disability.csv", sep = '\t', header = TRUE)
pos_train_names <- pos_table[,1]
# Negative genes
neg_table <- read.csv(file = "data/control.csv", sep = '\t', header = TRUE)
neg_train_names <- neg_table[,1]

a <- listMarts()
ensembl=useMart("ensembl")
datasets <- listDatasets(ensembl)

ensembl = useMart("ensembl",dataset="hsapiens_gene_ensembl")

for(gene in pos_train_names){
  bm <- getBM(attributes=c('ensembl_gene_id', 'ensembl_transcript_id', 'coding'),
              filters = c('external_gene_name'),
              values = gene,
              mart = ensembl)
  
  biomaRt::exportFASTA(bm, paste(c('data/pos_seqs/', gene, '.fasta'), collapse = ""))
}

for(gene in neg_train_names){
 bm <- getBM(attributes=c('ensembl_gene_id', 'ensembl_transcript_id', 'coding'),
    filters = c('external_gene_name'),
   values = gene,
  mart = ensembl)

  biomaRt::exportFASTA(bm, paste(c('data/neg_seqs/', gene, '.fasta'), collapse = ""))
}