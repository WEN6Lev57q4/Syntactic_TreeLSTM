import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import RNNDropout, Seq2SeqEncoder, SoftmaxAttention
from utils import get_mask, replace_masked
import math
# module for distance-angle similarity
class Similarity(nn.Module):
    def __init__(self,  hidden_dim, num_classes):
        super(Similarity, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.wh = nn.Linear(4*hidden_dim, self.hidden_dim)
        self.wp = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, lvec, rvec):
        # lvec = torch.unsqueeze(lvec,0)
        # rvec = torch.unsqueeze(rvec,0)
        max_lvec = F.max_pool2d(lvec, kernel_size=(lvec.size(1),1)).view(lvec.size(0),lvec.size(2))
        max_rvec = F.max_pool2d(rvec, kernel_size=(rvec.size(1),1)).view(rvec.size(0),rvec.size(2))
        avg_lvec = F.avg_pool2d(lvec, kernel_size=(lvec.size(1),1)).view(lvec.size(0),lvec.size(2))
        avg_rvec = F.avg_pool2d(rvec, kernel_size=(rvec.size(1),1)).view(rvec.size(0),rvec.size(2))
        v = torch.cat([max_lvec,max_rvec,avg_lvec,avg_rvec],dim=1)

        out = F.relu(self.wh(v))
        out = self.wp(out)
        return out


class TreeLSTMforNLI(nn.Module):
    def __init__(self, vocab_size, in_dim, mem_dim, hidden_dim, num_classes, sparsity, freeze):
        super(TreeLSTMforNLI, self).__init__()
        self.emb = nn.Embedding(vocab_size, in_dim, padding_idx=0, sparse=sparsity)
        if freeze:
            self.emb.weight.requires_grad = False
        self.treelstm = TreeLSTM(in_dim, mem_dim)
        self.gcn = GCN(mem_dim,mem_dim)
        self.similarity = Similarity(mem_dim,num_classes)

    def forward(self, linputs,lgraph, rinputs,rgraph,device):
        linputs = self.emb(linputs)
        rinputs = self.emb(rinputs)
        lstate, lhidden = self.treelstm(linputs,lgraph,device)
        rstate, rhidden = self.treelstm(rinputs,rgraph,device)
        lstate = self.gcn(lgraph,lstate,device)
        rstate = self.gcn(rgraph,rstate,device)
        output = self.similarity(lstate, rstate)
        return output

class GCN(nn.Module):
    def __init__(self,idim,odim):
        super(GCN,self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(idim, odim))
        self.bias = nn.Parameter(torch.FloatTensor(odim))
        self.reset_parameters()
    def forward(self, graph,h,device):
        time_step = graph.size(1)
        batch_size = graph.size(0)
        lower_triangular_matrix = torch.transpose(graph,1,2)
        Eye = torch.unsqueeze(torch.torch.eye(time_step),0).repeat(batch_size,1,1).to(device)
        graph = Eye+lower_triangular_matrix+graph

        output = F.relu(torch.bmm(graph,torch.matmul(h,self.weight)))

        return output

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)


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
        # self.wi = nn.Linear(idim,hdim)

    def forward(self,input,graph,device):
        # print(input.size())
        # print(graph.size())
        # input = torch.unsqueeze(input,0)
        # graph = torch.unsqueeze(graph,0)

        time_step = input.size(1)
        batch_size = input.size(0)
        # print(time_step)
        h = torch.zeros(batch_size,time_step,self.hdim).to(device)
        c = torch.zeros(batch_size,time_step,self.hdim).to(device)
        for j in torch.range(0,time_step-1):
            j = j.long()
            # step_mask = torch.zeros((batch_size,time_step,time_step)).to(device)
            # step_mask[:,j,j] = 1#(t*t)
            # x = torch.bmm(step_mask,input)#(t*D)
            x = torch.unsqueeze(input[:,j,:],1)
            # print('x',x)
            mask = torch.unsqueeze(graph[:,j,:],1)
            # mask = torch.bmm(step_mask,graph)#(t*t)
            # h_ = torch.bmm(mask,h)#(t*t)
            h_ = torch.unsqueeze(h[:,j,:],1)
            # print('h_',h_)
            i = F.sigmoid(self.wi(x)+self.ui(h_))
            # i = torch.bmm(step_mask,i)
            # print('i',i)
            o = F.sigmoid(self.wo(x)+self.uo(h_))
            # o = torch.bmm(step_mask,o)
            # print('o',o)
            u = F.tanh(self.wu(x)+self.uu(h_))
            del h_
            # u = torch.bmm(step_mask,u)#(t*)
            # print('u',u)#(1*D)
            x = x.repeat(1,time_step,1)
            f = F.sigmoid(self.wf(x)+self.uf(h))
            c_ = i * u
            del i ,u,x
            f = f*c
            c_ = torch.bmm(mask,f)+ c_
            del f
            o_ = o * F.tanh(c_)
            del o,mask

            if j==0:
                c = torch.cat([c_,c[:,1:,:]],1)
                h = torch.cat([o_,h[:,1:,:]],1)
            elif j==time_step-1:
                c = torch.cat([c[:,:j,:],c_],1)
                h = torch.cat([h[:,:j,:],o_],1)
            else:
                c = torch.cat([c[:,:j,:],c_,c[:,j+1:,:]],1)
                h = torch.cat([h[:,:j,:],o_,h[:,j+1:,:]],1)
        return h,c

class ESIM(nn.Module):
    """
    Implementation of the ESIM model presented in the paper "Enhanced LSTM for
    Natural Language Inference" by Chen et al.
    """

    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 hidden_size,
                 embeddings=None,
                 padding_idx=0,
                 dropout=0.5,
                 num_classes=3,
                 device="cpu",
                 freeze=True,
                 ):
        """
        Args:
            vocab_size: The size of the vocabulary of embeddings in the model.
            embedding_dim: The dimension of the word embeddings.
            hidden_size: The size of all the hidden layers in the network.
            embeddings: A tensor of size (vocab_size, embedding_dim) containing
                pretrained word embeddings. If None, word embeddings are
                initialised randomly. Defaults to None.
            padding_idx: The index of the padding token in the premises and
                hypotheses passed as input to the model. Defaults to 0.
            dropout: The dropout rate to use between the layers of the network.
                A dropout rate of 0 corresponds to using no dropout at all.
                Defaults to 0.5.
            num_classes: The number of classes in the output of the network.
                Defaults to 3.
            device: The name of the device on which the model is being
                executed. Defaults to 'cpu'.
        """
        super(ESIM, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout = dropout
        self.device = device

        self.emb = nn.Embedding(self.vocab_size,
                                            self.embedding_dim,
                                            padding_idx=padding_idx,
                                            _weight=embeddings)

        if freeze:
            self.emb.weight.requires_grad = False

        if self.dropout:
            self._rnn_dropout = RNNDropout(p=self.dropout)
            # self._rnn_dropout = nn.Dropout(p=self.dropout)

        # self._encoding = Seq2SeqEncoder(nn.LSTM,
        #                                 self.embedding_dim,
        #                                 self.hidden_size,
        #                                 bidirectional=True)
        self._encoding = TreeLSTM(self.embedding_dim,self.hidden_size)

        self._attention = SoftmaxAttention()

        self._projection = nn.Sequential(nn.Linear(4*self.hidden_size,
                                                   self.hidden_size),
                                         nn.ReLU())

        # self._composition = Seq2SeqEncoder(nn.LSTM,
        #                                    self.hidden_size,
        #                                    self.hidden_size,
        #                                    bidirectional=True)
        self._composition = TreeLSTM(self.hidden_size,self.hidden_size)
        self._classification = nn.Sequential(nn.Dropout(p=self.dropout),
                                             nn.Linear(4*self.hidden_size,
                                                       self.hidden_size),
                                             nn.Tanh(),
                                             nn.Dropout(p=self.dropout),
                                             nn.Linear(self.hidden_size,
                                                       self.num_classes))

        # Initialize all weights and biases in the model.
        self.apply(_init_esim_weights)

    def forward(self,
                premises,
                premises_lengths,
                hypotheses,
                hypotheses_lengths,lgraph,rgraph):
        """
        Args:
            premises: A batch of varaible length sequences of word indices
                representing premises. The batch is assumed to be of size
                (batch, premises_length).
            premises_lengths: A 1D tensor containing the lengths of the
                premises in 'premises'.
            hypothesis: A batch of varaible length sequences of word indices
                representing hypotheses. The batch is assumed to be of size
                (batch, hypotheses_length).
            hypotheses_lengths: A 1D tensor containing the lengths of the
                hypotheses in 'hypotheses'.
        Returns:
            logits: A tensor of size (batch, num_classes) containing the
                logits for each output class of the model.
            probabilities: A tensor of size (batch, num_classes) containing
                the probabilities of each output class in the model.
        """
        premises_mask = get_mask(premises, premises_lengths).to(self.device)
        hypotheses_mask = get_mask(hypotheses, hypotheses_lengths)\
            .to(self.device)

        embedded_premises = self.emb(premises)
        embedded_hypotheses = self.emb(hypotheses)

        if self.dropout:
            embedded_premises = self._rnn_dropout(embedded_premises)
            embedded_hypotheses = self._rnn_dropout(embedded_hypotheses)

        encoded_premises,_ = self._encoding(embedded_premises,lgraph,self.device
                                          )
        encoded_hypotheses,_ = self._encoding(embedded_hypotheses,rgraph,self.device
                                            )

        attended_premises, attended_hypotheses =\
            self._attention(encoded_premises, premises_mask,
                            encoded_hypotheses, hypotheses_mask)

        enhanced_premises = torch.cat([encoded_premises,
                                       attended_premises,
                                       encoded_premises - attended_premises,
                                       encoded_premises * attended_premises],
                                      dim=-1)
        enhanced_hypotheses = torch.cat([encoded_hypotheses,
                                         attended_hypotheses,
                                         encoded_hypotheses -
                                         attended_hypotheses,
                                         encoded_hypotheses *
                                         attended_hypotheses],
                                        dim=-1)

        projected_premises = self._projection(enhanced_premises)
        projected_hypotheses = self._projection(enhanced_hypotheses)

        if self.dropout:
            projected_premises = self._rnn_dropout(projected_premises)
            projected_hypotheses = self._rnn_dropout(projected_hypotheses)

        v_ai,_ = self._composition(projected_premises, lgraph,self.device)
        v_bj,_ = self._composition(projected_hypotheses, rgraph,self.device)

        v_a_avg = torch.sum(v_ai * premises_mask.unsqueeze(1)
                                                .transpose(2, 1), dim=1)\
            / torch.sum(premises_mask, dim=1, keepdim=True)
        v_b_avg = torch.sum(v_bj * hypotheses_mask.unsqueeze(1)
                                                  .transpose(2, 1), dim=1)\
            / torch.sum(hypotheses_mask, dim=1, keepdim=True)

        v_a_max, _ = replace_masked(v_ai, premises_mask, -1e7).max(dim=1)
        v_b_max, _ = replace_masked(v_bj, hypotheses_mask, -1e7).max(dim=1)

        v = torch.cat([v_a_avg, v_a_max, v_b_avg, v_b_max], dim=1)

        logits = self._classification(v)

        return logits


def _init_esim_weights(module):
    """
    Initialise the weights of the ESIM model.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)
        nn.init.constant_(module.bias.data, 0.0)

    elif isinstance(module, nn.LSTM):
        nn.init.xavier_uniform_(module.weight_ih_l0.data)
        nn.init.orthogonal_(module.weight_hh_l0.data)
        nn.init.constant_(module.bias_ih_l0.data, 0.0)
        nn.init.constant_(module.bias_hh_l0.data, 0.0)
        hidden_size = module.bias_hh_l0.data.shape[0] // 4
        module.bias_hh_l0.data[hidden_size:(2*hidden_size)] = 1.0

        if (module.bidirectional):
            nn.init.xavier_uniform_(module.weight_ih_l0_reverse.data)
            nn.init.orthogonal_(module.weight_hh_l0_reverse.data)
            nn.init.constant_(module.bias_ih_l0_reverse.data, 0.0)
            nn.init.constant_(module.bias_hh_l0_reverse.data, 0.0)
            module.bias_hh_l0_reverse.data[hidden_size:(2*hidden_size)] = 1.0

if __name__ == '__main__':
    pass
