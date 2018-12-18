#!/usr/bin/python
import sys
import os
import numpy
import pickle as pkl
import re

from collections import OrderedDict

dic = {'entailment': '0', 'neutral': '1', 'contradiction': '2'}

def build_dictionary(filepaths, dst_path, lowercase=False):
    word_freqs = OrderedDict()
    for filepath in filepaths:
        print('Processing', filepath)
        with open(filepath, 'r') as f:
            for line in f:
                if lowercase:
                    line = line.lower()

                line = re.sub('\\([A-Z|.]+', '(', line)
                line = re.sub('\\(', '( ', line)
                line = re.sub('\\)', '', line)
                line = re.sub('[ ]+', ' ', line)
                words_in = line.strip().split(' ')

                for w in words_in:
                    if w not in word_freqs:
                        word_freqs[w] = 0
                    word_freqs[w] += 1

    words = list(word_freqs.keys())
    freqs = list(word_freqs.values())

    sorted_idx = numpy.argsort(freqs)
    sorted_words = [words[ii] for ii in sorted_idx[::-1]]

    worddict = OrderedDict()
    worddict['_PAD_'] = 0 # default, padding 
    worddict['_UNK_'] = 1 # out-of-vocabulary
    worddict['_BOS_'] = 2 # begin of sentence token
    worddict['_EOS_'] = 3 # end of sentence token

    for ii, ww in enumerate(sorted_words):
        worddict[ww] = ii + 4
    #
    # with open(dst_path, 'wb') as f:
    #     pkl.dump(worddict, f)
    with open(dst_path, 'w') as f:
        for w in worddict.keys():
            f.write(w + '\n')
    for k,v in worddict.items():
        print(k,v)
    print('Dict size', len(worddict))
    print('Done')


def build_dtree_sequence(filepath, dst_dir):
    filename = os.path.basename(filepath)
    print('*'*80)
    print("DTree file:",filename)
    len_p = []
    len_h = []
    premises= []
    hypothesis = []
    labels = []
    with open(filepath) as f:
        next(f) # skip the header row
        for line in f:
            sents = line.strip().split('\t')
            if sents[0] is '-':
                continue

            words_in = sents[3].strip()
            # words_in = [x for x in words_in if x not in ('(',')')]
            # f1.write(words_in+ '\n')
            premises.append(words_in)
            # len_p.append(len(words_in))

            words_in = sents[4].strip()
            # words_in = [x for x in words_in if x not in ('(',')')]
            # f2.write(words_in+ '\n')
            hypothesis.append(words_in)
            # len_p.append(len(words_in))

            # f3.write(dic[sents[0]] + '\n')

            labels.append(dic[sents[0]])


    sample_num = len(premises)
    print('Sample Num:',sample_num)
    numpy.random.seed(123)
    indices = numpy.arange(sample_num)
    numpy.random.shuffle(indices)
    print(indices[:10])
    with open(os.path.join(dst_dir, 'premise_%s' % filename), 'w') as f1,\
        open(os.path.join(dst_dir, 'hypothesis_%s' % filename), 'w') as f2, \
            open(os.path.join(dst_dir, 'label_%s' % filename),'w') as f3:
        for  i in indices:
            f1.write(premises[i]+'\n')
            f2.write(hypothesis[i]+'\n')
            f3.write(labels[i]+'\n')
    # print('max min len premise', max(len_p), min(len_p))
    # print('max min len hypothesis', max(len_h), min(len_h))

def build_ctree_sequence(filepath, dst_dir):
    filename = os.path.basename(filepath)
    print('*'*80)
    print("CTree file:",filename)
    len_p = []
    len_h = []
    with open(filepath) as f, \
         open(os.path.join(dst_dir, 'premise_%s'%filename), 'w') as f1, \
         open(os.path.join(dst_dir, 'hypothesis_%s'%filename), 'w') as f2,  \
         open(os.path.join(dst_dir, 'label_%s'%filename), 'w') as f3:
        next(f) # skip the header row
        for line in f:
            sents = line.strip().split('\t')
            if sents[0] is '-':
                continue

            words_in = sents[1].strip().split(' ')
            words_in = [x for x in words_in if x not in ('(',')')]
            f1.write(' '.join(words_in) + '\n')
            len_p.append(len(words_in))

            words_in = sents[2].strip().split(' ')
            words_in = [x for x in words_in if x not in ('(',')')]
            f2.write(' '.join(words_in) + '\n')
            len_h.append(len(words_in))

            f3.write(dic[sents[0]] + '\n')

    print('max min len premise', max(len_p), min(len_p))
    print('max min len hypothesis', max(len_h), min(len_h))




def make_dirs(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

if __name__ == '__main__':
    print('=' * 80)
    print('Preprocessing snli_1.0 dataset')
    print('=' * 80)
    base_dir = os.path.dirname(os.path.realpath(__file__))
    dst_tree_dir = os.path.join(base_dir, 'dtree')
    dst_ctree_dir = os.path.join(base_dir, 'ctree')
    snli_dir = os.path.join(base_dir, 'snli/snli_1.0')
    make_dirs([dst_tree_dir,dst_ctree_dir])

    build_dtree_sequence(os.path.join(snli_dir, 'snli_1.0_dev.txt'), dst_tree_dir)
    build_dtree_sequence(os.path.join(snli_dir, 'snli_1.0_test.txt'), dst_tree_dir)
    build_dtree_sequence(os.path.join(snli_dir, 'snli_1.0_train.txt'), dst_tree_dir)

    build_ctree_sequence(os.path.join(snli_dir, 'snli_1.0_dev.txt'), dst_ctree_dir)
    build_ctree_sequence(os.path.join(snli_dir, 'snli_1.0_test.txt'), dst_ctree_dir)
    build_ctree_sequence(os.path.join(snli_dir, 'snli_1.0_train.txt'), dst_ctree_dir)

    build_dictionary([os.path.join(dst_tree_dir, 'premise_snli_1.0_train.txt'),
                      os.path.join(dst_tree_dir, 'hypothesis_snli_1.0_train.txt')],
                      os.path.join(dst_tree_dir, 'snli_vocab_cased.txt'))

