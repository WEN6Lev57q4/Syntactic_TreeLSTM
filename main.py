import numpy as np
import os
import torch
from torch.autograd import Variable
import torch.nn
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn
import torch.optim as optim
from dataset.dataset import NLIdataset
from dataset.vocab import Vocab
from trainer import Trainer
from config import parse_args
from model.model import TreeLSTMforNLI,ESIM
import utils


def main():
    global args
    args = parse_args()
    vocab_file = os.path.join(args.dtree, 'snli_vocab_cased.txt')
    vocab = Vocab(filename=vocab_file)

    args.cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if args.cuda else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    l_train_file = os.path.join(args.dtree,args.premise_train)
    r_train_file = os.path.join(args.dtree,args.hypothesis_train)
    label_train_file = os.path.join(args.dtree,args.label_train)

    l_dev_file = os.path.join(args.dtree,args.premise_dev)
    r_dev_file = os.path.join(args.dtree,args.hypothesis_dev)
    label_dev_file = os.path.join(args.dtree,args.label_dev)

    l_test_file = os.path.join(args.dtree,args.premise_test)
    r_test_file = os.path.join(args.dtree,args.hypothesis_test)
    label_test_file = os.path.join(args.dtree,args.label_test)

    l_train_squence_file = os.path.join(args.ctree,args.premise_train)
    r_train_squence_file = os.path.join(args.ctree,args.hypothesis_train)

    l_dev_squence_file = os.path.join(args.ctree,args.premise_dev)
    r_dev_squence_file = os.path.join(args.ctree,args.hypothesis_dev)

    l_test_squence_file = os.path.join(args.ctree,args.premise_test)
    r_test_squence_file = os.path.join(args.ctree,args.hypothesis_test)

    print(l_train_file,l_dev_file,l_test_file)
    print(r_train_file,r_dev_file,r_test_file)
    print(label_train_file,label_dev_file,label_test_file)


    # load SICK dataset splits
    train_file = os.path.join(args.data, 'train.pth')
    if os.path.isfile(train_file):
        train_dataset = torch.load(train_file)
    else:
        train_dataset = NLIdataset(premise_tree=l_train_file, hypothesis_tree=r_train_file,
                                   premise_seq=l_train_squence_file, hypothesis_seq=r_train_squence_file,
                                   label=label_train_file, vocab=vocab, num_classes=3,args=args)
        torch.save(train_dataset, train_file)
    if args.savedev ==1 :
        dev_file = os.path.join(args.data, 'dev.pth')
        if os.path.isfile(dev_file):
            dev_dataset = torch.load(dev_file)
        else:
            dev_dataset = NLIdataset(premise_tree=l_dev_file, hypothesis_tree=r_dev_file,
                                   premise_seq=l_dev_squence_file, hypothesis_seq=r_dev_squence_file,
                                   label=label_dev_file, vocab=vocab, num_classes=3,args=args)
            torch.save(dev_dataset, dev_file)

        test_file = os.path.join(args.data, 'test.pth')
        if os.path.isfile(test_file):
            test_dataset = torch.load(test_file)
        else:
            test_dataset = NLIdataset(premise_tree=l_test_file, hypothesis_tree=r_test_file,
                                       premise_seq=l_test_squence_file, hypothesis_seq=r_test_squence_file,
                                       label=label_test_file, vocab=vocab, num_classes=3,args=args)
            torch.save(test_dataset, test_file)
    else:
        dev_dataset = NLIdataset(premise_tree=l_dev_file, hypothesis_tree=r_dev_file,
                                   premise_seq=l_dev_squence_file, hypothesis_seq=r_dev_squence_file,
                                   label=label_dev_file, vocab=vocab, num_classes=3,args=args)
        test_dataset = NLIdataset(premise_tree=l_test_file, hypothesis_tree=r_test_file,
                                       premise_seq=l_test_squence_file, hypothesis_seq=r_test_squence_file,
                                       label=label_test_file, vocab=vocab, num_classes=3,args=args)

    train_data_loader = DataLoader(train_dataset,batch_size=args.batchsize,shuffle=False)
    dev_data_loader = DataLoader(dev_dataset,batch_size=args.batchsize,shuffle=False)
    test_data_loader = DataLoader(test_dataset,batch_size=args.batchsize,shuffle=False)

    # for data in train_data_loader:
    #     lsent, lgraph, rsent, rgraph, label = data
    #     print(label)
    #     break

    # # initialize model, criterion/loss_function, optimizer
    # model = TreeLSTMforNLI(
    #     vocab.size(),
    #     args.input_dim,
    #     args.mem_dim,
    #     args.hidden_dim,
    #     args.num_classes,
    #     args.sparse,
    #     args.freeze_embed)


    # for words common to dataset vocab and GLOVE, use GLOVE vectors
    # for other words in dataset vocab, use random normal vectors
    emb_file = os.path.join(args.data, 'snli_embed.pth')
    if os.path.isfile(emb_file):
        emb = torch.load(emb_file)
    else:
    # load glove embeddings and vocab
        glove_vocab, glove_emb = utils.load_word_vectors(
            os.path.join(args.glove, 'glove.840B.300d'))
        emb = torch.zeros(vocab.size(), glove_emb.size(1), dtype=torch.float, device=device)
        emb.normal_(0, 0.05)
        # zero out the embeddings for padding and other special words if they are absent in vocab
        for idx, item in enumerate(['_PAD_','_UNK_',
            '_BOS_', '_EOS_']):
            emb[idx].zero_()
        for word in vocab.labelToIdx.keys():
            if glove_vocab.getIndex(word):
                emb[vocab.getIndex(word)] = glove_emb[glove_vocab.getIndex(word)]
        torch.save(emb, emb_file)
    # plug these into embedding matrix inside model
    # model.emb.weight.data.copy_(emb)
    model = ESIM(vocab.size(),
                 args.input_dim,
                 args.mem_dim,
                 embeddings=emb,
                 dropout=0.5,
                 num_classes=args.num_classes,
                 device=device,
                 freeze=args.freeze_embed).to(device)
    criterion = nn.CrossEntropyLoss()
    model.to(device), criterion.to(device)
    if args.optim == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr = args.lr, weight_decay = args.wd)
    elif args.optim == 'adagrad':
        optimizer = optim.Adagrad(filter(lambda p: p.requires_grad,model.parameters()), lr = args.lr, weight_decay = args.wd)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad,model.parameters()), lr = args.lr, weight_decay = args.wd)

    trainer = Trainer(args, model, criterion, optimizer, device)

    best = -999.0
    best_loop = 0
    for epoch in range(args.epochs):
        train_loss = trainer.train(train_data_loader)
        train_loss, train_acc = trainer.test(train_data_loader)
        dev_loss, dev_acc = trainer.test(dev_data_loader)
        test_loss, test_acc = trainer.test(test_data_loader)

        print('==> Epoch {}, Train \tLoss: {}\tAcc: {}'.format(epoch, train_loss, train_acc))
        print('==> Epoch {}, Dev \tLoss: {}\tAcc: {}'.format(epoch, dev_loss, dev_acc))
        print('==> Epoch {}, Test \tLoss: {}\tAcc: {}'.format(epoch, test_loss, test_acc))

        if best < test_acc:
            best = test_acc
            best_loop = 0
            print('Get Improvement,Save Model, The best performence is %f'%(best))
            checkpoint = {
                'model': trainer.model.state_dict(),
                'optim': trainer.optimizer,
                'acc': test_acc,
                'args': args, 'epoch': epoch
            }
            print('==> New optimum found, checkpointing everything now...')
            torch.save(checkpoint, '%s.pt' % os.path.join(args.save, args.expname))
        else:
            best_loop+=1
            if best_loop > args.patience:
                print('Early Stop,Best Acc:%f'%(best))
                break


if __name__ == '__main__':
    main()