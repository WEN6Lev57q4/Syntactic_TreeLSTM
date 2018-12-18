import os,sys
from tqdm import tqdm
from copy import deepcopy
from dataset.tree import Tree
import torch
import torch.utils.data as data
import re
import numpy as np
import random

import math
# Dataset class for NLI dataset
class NLIdataset(data.Dataset):
    def __init__(self, premise_tree,hypothesis_tree,premise_seq,hypothesis_seq,label, vocab, num_classes,args):
        super(NLIdataset, self).__init__()
        self.args = args
        self.vocab = vocab
        self.num_classes = num_classes
        self.batch_size = args.batchsize
        self.sample_num  = 10000000
        self.lgraphs,self.lsentences,self.llength = self.read_trees(premise_tree)
        self.rgraphs,self.rsentences,self.rlength = self.read_trees(hypothesis_tree)
        self.labels = self.read_labels(label)
        # self.premise_length = max(self.llength)
        # self.hypothesis_length = max(self.rlength)
        # self.shuffle_list()
        self.convert_to_tensor(self.lgraphs,self.lsentences,self.llength)
        self.convert_to_tensor(self.rgraphs,self.rsentences,self.rlength)
        # self.convert_to_seq_l(self.lsentences,self.llength,self.premise_length)
        # self.convert_to_seq_l(self.rsentences,self.rlength,self.hypothesis_length)
        self.size = len(self.labels)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        lsent = self.lsentences[index]
        rsent = self.rsentences[index]
        llen =  self.llength[index]
        rlen =  self.rlength[index]
        lgraph =self.lgraphs[index]
        rgraph = self.rgraphs[index]
        label = self.labels[index]
        return (lsent,lgraph,llen,rsent,rgraph,rlen,label)
        # return (lsent,llen,rsent,rlen,label)

    def shuffle_list(self):
        random.seed(self.args.seed)
        random.shuffle(self.labels)
        random.shuffle(self.lsentences)
        random.shuffle(self.lgraphs)
        random.shuffle(self.llength)
        random.shuffle(self.rsentences)
        random.shuffle(self.rgraphs)
        random.shuffle(self.rlength)

    def read_sentences(self, filename):
        with open(filename, 'r') as f:
            sentences = [self.read_sentence(line) for line in tqdm(f.readlines())]
        return sentences

    def read_sentence(self, line):
        indices = self.vocab.convertToIdx(line.split(), '_UNK_')
        # indices = indices
        l = len(indices)
        return indices,l

    def read_trees(self, filename):
        # trees  = []
        sentences = []
        graphs = []
        length = []
        with open(filename, 'r') as f:
            for line in tqdm(f.readlines()):
                graph,sequence,l= self.read_tree(line)
                # trees.append(tree)
                graphs.append(graph)
                sentences.append(sequence)
                length.append(l)
        if self.sample_num > len(sentences):
            self.sample_num = len(sentences)

        return graphs[:self.sample_num],sentences[:self.sample_num],length[:self.sample_num]

    def convert_to_tensor(self,graphs,sentences,length):
        start = 0
        end = start+self.batch_size
        LENGTH = len(self.labels)
        while start<=LENGTH:
            # print(start,end,'='*10)
            if start==end:
                max_length = length[LENGTH-1]
            else:
                max_length = max(length[start:end])
            for i in range(start,end,1):
                graph = np.zeros((max_length,max_length))
                graph_length = graphs[i].shape[0]
                graph[:graph_length,:graph_length] = graphs[i]
                graphs[i] = torch.from_numpy(graph).float()

                sentences[i] = sentences[i]+(max_length-len(sentences[i]))*[0]
                sentences[i] = torch.LongTensor(sentences[i])
                # print(i)
            # Batch.append((torch.cat(graphs[start:end],dim=0),torch.cat(sentences[start:end],dim=0),length[start:end]))
            start = start+self.batch_size
            if end+self.batch_size >= LENGTH:
                end = LENGTH
            else:
                end = end+self.batch_size
        return

    def convert_to_seq_l(self,sentences,length,max_length):
        start = 0
        end = start+self.batch_size
        LENGTH = len(self.labels)
        while start<=LENGTH:
            # print(start,end,'='*10)
            for i in range(start,end,1):
                graph = np.zeros((max_length,max_length))
                # graph_length = graphs[i].shape[0]
                # graph[:graph_length,:graph_length] = graphs[i]
                # graphs[i] = torch.from_numpy(graph).float()

                sentences[i] = sentences[i]+(max_length-len(sentences[i]))*[0]
                sentences[i] = torch.LongTensor(sentences[i])
                # print(i)
            # Batch.append((torch.cat(graphs[start:end],dim=0),torch.cat(sentences[start:end],dim=0),length[start:end]))
            start = start+self.batch_size
            if end+self.batch_size >= LENGTH:
                end = LENGTH
            else:
                end = end+self.batch_size
        return

    def read_tree(self, line):

        line = line.rstrip()
        line = re.sub('\\([A-Z|.]+','(',line)
        line = re.sub('\\(', '( ', line)
        line = re.sub('\\)', ' )', line)
        line = re.sub('[ ]+',' ',line)

        parts = line.split()
        stack = []
        tree = None
        index = 0
        for p_i, part in enumerate(parts):

            if part == '(':  #####父节点，不见 ）推入栈
                if tree is None:
                    tree = Tree(index)
                elif parts[p_i + 1] != '(':
                    continue
                else:
                    tree.add_children(stack[-1], index, '(')

                stack.append(index)
                index += 1
            elif part == ')' and parts[p_i - 1] == ')':
                stack.pop(-1)
            elif part != ')':
                tree.add_children(stack[-1], index, part)
                index += 1
        tree.resort_sequence()
        sequence = tree.return_sequence()
        sequence,l = self.read_sentence(sequence)
        graph = tree.convert_to_graph()
        # mask_graph = np.zeros((self.max_length,self.max_length))
        # mask_graph[:l,:l] = graph
        # for i in range(l):
            # mask_graph[i,:l] = graph[i]
        return graph,sequence,l

    def read_labels(self, filename):
        with open(filename, 'r') as f:
            labels = list(map(lambda x: float(x), f.readlines()))
            labels = torch.tensor(labels, dtype=torch.long, device='cpu')
            if self.sample_num >  len(labels):
                self.sample_num = len(labels)
        return labels[:self.sample_num]
if __name__ == '__main__':
    pass