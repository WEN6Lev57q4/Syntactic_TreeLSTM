import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='PyTorch TreeLSTM for Sentence Similarity on Dependency Trees')
    # data arguments
    parser.add_argument('--data', default='data/snli/',
                        help='path to dataset')
    parser.add_argument('--glove', default='data/glove/',
                        help='directory with GLOVE embeddings')
    parser.add_argument('--save', default='checkpoints/',
                        help='directory to save checkpoints in')
    parser.add_argument('--expname', type=str, default='1test1',
                        help='Name to identify experiment')
    parser.add_argument('--vocab',default='data/ctree/snli_vocab_cased.pkl',
                        help='vocab file path'),
    parser.add_argument('--ctree',default='data/ctree',
                        help='constituency tree file'),
    parser.add_argument('--dtree',default='data/dtree',
                        help='dependency tree file'),
    parser.add_argument('--premise_train',default='_'.join(['premise','snli_1.0','train.txt']))
    parser.add_argument('--hypothesis_train',default='_'.join(['hypothesis','snli_1.0','train.txt']))
    parser.add_argument('--label_train',default='_'.join(['label','snli_1.0','train.txt']))

    parser.add_argument('--premise_dev',default='_'.join(['premise','snli_1.0','dev.txt']))
    parser.add_argument('--hypothesis_dev',default='_'.join(['hypothesis','snli_1.0','dev.txt']))
    parser.add_argument('--label_dev',default='_'.join(['label','snli_1.0','dev.txt']))

    parser.add_argument('--premise_test',default='_'.join(['premise','snli_1.0','test.txt']))
    parser.add_argument('--hypothesis_test',default='_'.join(['hypothesis','snli_1.0','test.txt']))
    parser.add_argument('--label_test',default='_'.join(['label','snli_1.0','test.txt']))

    # model arguments
    parser.add_argument('--input_dim', default=300, type=int,
                        help='Size of input word vector')
    parser.add_argument('--mem_dim', default=300, type=int,
                        help='Size of TreeLSTM cell state')
    parser.add_argument('--hidden_dim', default=300, type=int,
                        help='Size of classifier MLP')
    parser.add_argument('--num_classes', default=3, type=int,
                        help='Number of classes in dataset')
    parser.add_argument('--freeze_embed', action='store_true',
                        help='Freeze word embeddings')
    parser.add_argument('--max_length',default=152,type=int,
                        help='Max tree length')
    # training arguments
    parser.add_argument('--epochs', default=15, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--batchsize', default=8, type=int,
                        help='batchsize for optimizer updates')
    parser.add_argument('--lr', default=0.001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--wd', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--sparse', action='store_true',
                        help='Enable sparsity for embeddings, \
                              incompatible with weight decay')
    parser.add_argument('--optim', default='adam',
                        help='optimizer (default: adagrad)')
    parser.add_argument('--patience',default='8',type=int,help='every stop epoch')
    # miscellaneous options
    parser.add_argument('--seed', default=123, type=int,
                        help='random seed (default: 123)')
    parser.add_argument('--savedev',default=1,type=int,help='wheather save testdata and devdata')
    cuda_parser = parser.add_mutually_exclusive_group(required=False)
    cuda_parser.add_argument('--cuda', dest='cuda', action='store_false')
    cuda_parser.add_argument('--no-cuda', dest='cuda', action='store_false')
    parser.set_defaults(cuda=True)

    args = parser.parse_args()
    return args
