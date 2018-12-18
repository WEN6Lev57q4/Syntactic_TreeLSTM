import numpy as np
from queue import Queue
import torch.optim as optim
import re
import torch
import torch.optim
from torch.autograd import Variable
import torch.nn as nn
from dataset.vocab import Vocab
from dataset.dataset import NLIdataset
from dataset.vocab import Vocab
from trainer import Trainer
from config import parse_args
import os
from torch.utils.data import DataLoader
from model.model import TreeLSTMforNLI
from model.model import ESIM
from tqdm import tqdm
import utils
import torch.nn.functional as F

line = '(ROOT (S (NP (NP (DT A) (NN person)) (PP (IN on) (NP (DT a) (NN horse)))) (VP (VBZ jumps) (PP (IN over) (NP (DT a) (JJ broken) (JJ down) (NN airplane)))) (. .)))'
line = re.sub('\\([A-Z|.]+', '(', line)
line = re.sub('\\(', '( ', line)
line = re.sub('\\)', ' )', line)
line = re.sub('[ ]+', ' ', line)
#

class Tree(object):
    def __init__(self,index):
        self.root =  None
        self.node_array = {}
        node = TreeNode(index,None)
        node.set_word('(')
        self.root = node
        self.node_array[0] = node

    def add_children(self,parent_index,index,word):
        node = TreeNode(index, parent_index)
        node.set_word(word)
        self.node_array[index] = node
        self.node_array[parent_index].add_children(node)
    def resort_sequence(self):
        new_array = []
        tree_to_bfs_to_reverse = []
        reverse_to_bfs_to_tree = {}
        q = Queue()
        q.put(self.root)
        index = 0
        while not q.empty():
            node = q.get()
            new_array.append(node)
            tree_to_bfs_to_reverse.append(node.index)
            index += 1
            children = node.children
            children.reverse()
            for child in children:
                q.put(child)

        new_array.reverse()
        tree_to_bfs_to_reverse.reverse()

        for reverse,tree in enumerate(tree_to_bfs_to_reverse):
            reverse_to_bfs_to_tree[tree]=reverse


        self.new_array = new_array
        self.reverse_to_bfs_to_tree =  reverse_to_bfs_to_tree
        return
   
    def return_sequence(self):
        sequence = ' '.join([node.get_word() for node in self.new_array])
        return sequence


    def convert_to_graph(self):
        node_num = len(self.node_array.keys())
        graph = np.zeros((node_num,node_num),dtype='float32')
        for parent_index,node in enumerate(self.new_array):
              for child in node.children:
                  child_index = self.reverse_to_bfs_to_tree[child.index]
                  graph[parent_index][child_index] = 1
        return graph

def print_tree(tree):

    if not tree.has_children():
        print(tree.get_word())
    else:
        for child in tree.children:
            print_tree(child)
        print(tree.get_word())
    return 0


class TreeNode(object):
    def __init__(self,index,parent):
        self.index = index
        self.parent = parent
        self.children = []
        self.word = None

    def add_children(self,Node):
        self.children.append(Node)

    def set_word(self,word):
        self.word = word

    def get_word(self):
        return self.word

    def has_children(self):
        return not len(self.children)== 0



parts = line.split()
stack = []
tree = None
index = 0
for p_i,part in enumerate(parts):

    if part =='(':#####父节点，不见 ）推入栈
        if tree is None:
            tree = Tree(index)
        elif parts[p_i+1]!='(':
            continue
        else:
            tree.add_children(stack[-1],index,'(')

        stack.append(index)
        index+=1
    elif part == ')' and parts[p_i-1]==')':
        stack.pop(-1)
    elif part != ')':
        tree.add_children(stack[-1],index,part)
        index+=1

# print(tree.resort_sequence())
# print(tree.return_sequence())
# print(parts)
# print_tree(tree.root)


# graph = tree.convert_to_graph()




# length = graph.shape[0]
# print(graph.shape)
# print(length)
graph = np.zeros((2,2),dtype='float32')
input = np.zeros((2,2),dtype='float32')

class TreeLSTM(nn.Module):
    def __init__(self,idim,hdim):
        super(TreeLSTM,self).__init__()
        self.idim = idim
        self.hdim = hdim
        self.wi = nn.Linear(self.idim,self.hdim)
        self.wf = nn.Linear(self.idim,self.hdim)
        self.wo = nn.Linear(self.idim,self.hdim)
        self.wu = nn.Linear(self.idim,self.hdim)

        self.ui = nn.Linear(self.hdim,self.hdim)
        self.uf = nn.Linear(self.hdim,self.hdim)
        self.uo = nn.Linear(self.hdim,self.hdim)
        self.uu = nn.Linear(self.hdim,self.hdim)
        # self.lstm = nn.LSTM()
        # self.wi = nn.Linear(idim,hdim)

    def forward(self,input,graph):
        print(input.size())
        print(graph.size())
        batch_size = input.size(0)
        time_step = input.size(1)
        h = torch.zeros(batch_size,time_step,self.hdim)
        c = torch.zeros(batch_size,time_step,self.hdim)
        for j in torch.range(0,time_step-1):
            j = j.long()
            x = input[:,j,:]
            x = torch.unsqueeze(x,1)#[32, 1, 2])
            # print('x',x)
            mask = graph[:,j,:]
            mask = torch.unsqueeze(mask,1)#[32, 1, 21])
            # print('mask',mask)
            h_ = torch.bmm(mask,h)#([32, 1, 21]) * (32,21,2) -> (32,1,2)
            # print('h_',h_)
            h_.backward()

            i = self.wi(x)+self.ui(h_)#F.sigmoid(self.wi(x)+self.ui(h_))#(32,1,2)

            # print('i',i)
            o = self.wo(x)+self.uo(h_)#F.sigmoid(self.wo(x)+self.uo(h_))#(32,1,2)
            # print('o',o)

            u = self.wu(x)+self.uu(h_)#F.tanh(self.wu(x)+self.uu(h_))#(32,1,2)
            # print('u',u)

            x_= x.repeat(1,time_step,1)#(32,21,2)
            x_ = self.wi(x_)#(32,21,2)->(32,21,2)
            # print('x_',x_)

            h__ = self.uf(h)#(32,21,2)->
            # print('h',h__)

            f = x_+h__ #F.sigmoid(x_+h_)#(32,21,2)->(32,21,2)
            # print(f)

            t = torch.bmm(mask, f * c) + i * u
            # print('t',t)
            c[:,j,:] = torch.squeeze(t)#(32,1,21)*(32,21,2)->(32,1,2)
            h[:,j,:] = torch.squeeze(o*t)
        return h,c


model = TreeLSTM(2,2)

# criterion = nn.CrossEntropyLoss()
input = torch.from_numpy(input)
graph = torch.from_numpy(graph)
# print(input.size())
# print(graph.size())
input = torch.unsqueeze(input,0).repeat(1,1,1)
graph = torch.unsqueeze(graph,0).repeat(1,1,1)
input,graph = Variable(input),Variable(graph)

global args
args = parse_args()
vocab_file = os.path.join(args.dtree, 'snli_vocab_cased.txt')
vocab = Vocab(filename=vocab_file)

# args.cuda = args.cuda and torch.cuda.is_available()
# device = torch.device("cuda:0" if args.cuda else "cpu")

l_dev_file = os.path.join(args.dtree, args.premise_dev)
r_dev_file = os.path.join(args.dtree, args.hypothesis_dev)
label_dev_file = os.path.join(args.dtree, args.label_dev)

l_dev_squence_file = os.path.join(args.ctree, args.premise_dev)
r_dev_squence_file = os.path.join(args.ctree, args.hypothesis_dev)

l_test_file = os.path.join(args.dtree, args.premise_test)
r_test_file = os.path.join(args.dtree, args.hypothesis_test)
label_test_file = os.path.join(args.dtree, args.label_test)

l_test_squence_file = os.path.join(args.ctree, args.premise_test)
r_test_squence_file = os.path.join(args.ctree, args.hypothesis_test)
# dev_dataset = NLIdataset(premise_tree=l_dev_file, hypothesis_tree=r_dev_file,
#                          premise_seq=l_dev_squence_file, hypothesis_seq=r_dev_squence_file,
#                          label=label_dev_file, vocab=vocab, num_classes=3, args=args)
# for i in dev_dataset:
test_file = os.path.join(args.data, 'test.pth')
# test_dataset = torch.load(test_file)

test_dataset = NLIdataset(premise_tree=l_test_file, hypothesis_tree=r_test_file,
                          premise_seq=l_test_squence_file, hypothesis_seq=r_test_squence_file,
                          label=label_test_file, vocab=vocab, num_classes=3, args=args)
# torch.save(test_dataset, test_file)
#     print(i)

args.cuda = args.cuda and torch.cuda.is_available()
device = torch.device("cuda:2" if args.cuda else "cpu")
loader = DataLoader(test_dataset,batch_size=args.batchsize,shuffle=False)

model = TreeLSTMforNLI(
    vocab.size(),
    args.input_dim,
    args.mem_dim,
    args.hidden_dim,
    args.num_classes,
    args.sparse,
    args.freeze_embed)
criterion = nn.CrossEntropyLoss()


emb_file = os.path.join(args.data, 'snli_embed.pth')
emb = torch.load(emb_file)
model.emb.weight.data.copy_(emb)

# model = ESIM(vocab.size(),
#              args.input_dim,
#              args.mem_dim,
#              embeddings=emb,
#              dropout=0.5,
#              num_classes=args.num_classes,
#              device=device,
#              freeze=True
#              ).to(device)


model.to(device), criterion.to(device)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd)
optimizer.zero_grad()
total_loss = 0.0
accuracy = 0
model.train()
tqdm_loader = tqdm(loader)
for data in tqdm_loader:
    # lsent, llen, rsent, rlen, label = data
    lsent, lgraph, llen, rsent, rgraph, rlen, label = data
    target = utils.map_label_to_target_classifcation(label,args.num_classes)
    linput, rinput = lsent.to(device), rsent.to(device)
    lgraph, rgraph = lgraph.to(device), rgraph.to(device)
    llen, rlen = llen.to(device), rlen.to(device)
    target = target.to(device)
    label = label.to(device)
    # output = model(linput, llen,rinput, rlen)
    output = model(linput, lgraph, rinput, rgraph,device)
    loss = criterion(output, target)
    loss.backward()
    total_loss+=loss.item()
    description = "Avg. batch proc. loss: {:.4f}" \
        .format(total_loss)
    pred = torch.squeeze(output.data.max(1, keepdim=True)[1])
    accuracy += pred.eq(label.data.view_as(pred)).cpu().sum().numpy()
    # print(accuracy)
    optimizer.step()
    optimizer.zero_grad()

print(accuracy*1.0/len(loader.dataset))
# print(total_loss)
#     print(lsent)
#     print(lgraph)
#     print(label)
#     break
# print(len(loader))
# print(len(loader.dataset))
# h = model(input,graph)
# print(h)

a = [1,2,3,4,5]
b = [6,7,8,9,10]
