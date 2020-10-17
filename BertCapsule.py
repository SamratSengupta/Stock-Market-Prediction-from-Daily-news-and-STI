import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
from torch.optim import Adam

from pytorch_transformers import *
from transformers import BertModel


class Caps_Layer(nn.Module):
    def __init__(self, input_dim_capsule, num_capsule, dim_capsule, \
                 routings, kernel_size,T_epsilon,batch_size, share_weights=True,
                 activation='default', **kwargs):
        super(Caps_Layer, self).__init__(**kwargs)

        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size 
        self.share_weights = share_weights
        self.T_epsilon = T_epsilon
       
        if activation == 'default':
            self.activation = self.squash
        else:
            self.activation = nn.ReLU(inplace=True)

        if self.share_weights:
            self.W = nn.Parameter(
                nn.init.xavier_normal_(torch.empty(1, input_dim_capsule, self.num_capsule * self.dim_capsule)))
        else:
            self.W = nn.Parameter(
                torch.randn(batch_size, input_dim_capsule, self.num_capsule * self.dim_capsule))  # 64 batch_size

    def forward(self, x):

        if self.share_weights:
            u_hat_vecs = torch.matmul(x, self.W)
        else:
            print('add later')

        batch_size = x.size(0)
        input_num_capsule = x.size(1)
        u_hat_vecs = u_hat_vecs.view((batch_size, input_num_capsule,self.num_capsule, self.dim_capsule))                                     
        u_hat_vecs = u_hat_vecs.permute(0, 2, 1, 3)  # (batch_size,num_capsule,input_num_capsule,dim_capsule)
        b = torch.zeros_like(u_hat_vecs[:, :, :, 0])  # (batch_size,num_capsule,input_num_capsule)

        for i in range(self.routings):
            b = b.permute(0, 2, 1)
            
            c = F.softmax(b, dim=2)
            c = c.permute(0, 2, 1)
            b = b.permute(0, 2, 1)
            outputs = self.activation(torch.einsum('bij,bijk->bik', (c, u_hat_vecs)),self.T_epsilon)  # batch matrix multiplication
            # outputs shape (batch_size, num_capsule, dim_capsule)
            if i < self.routings - 1:
                b = torch.einsum('bik,bijk->bij', (outputs, u_hat_vecs))  # batch matrix multiplication
        return outputs  # (batch_size, num_capsule, dim_capsule)

    # text version of squash, slight different from original one
    def squash(self, x,T_epsilon, axis=-1):
        s_squared_norm = (x ** 2).sum(axis, keepdim=True)
        scale = torch.sqrt(s_squared_norm + T_epsilon)
        return x / scale
    
    
    
class Dense_Layer(nn.Module):
    def __init__(self,dropout_p,num_capsule,dim_capsule):
        super(Dense_Layer, self).__init__()        
        
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout_p, inplace=True),
            nn.Linear(num_capsule * dim_capsule, 1) 
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        return self.fc(x)
    
    
    
class Linear(nn.Module):
    def __init__(self, fan_in, fan_out):
        super(Linear, self).__init__()

        self.linear = nn.Linear(fan_in, fan_out)

        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)
    
class bertCapsuleModel(nn.Module):

    def __init__(self, input_dim_capsule,num_capsule, dim_capsule, 
                 routings, kernel_size,dropout_p,T_epsilon,batch_size,freeze_bert = False):
        
        super(bertCapsuleModel, self).__init__()
                                              
        self.bert = BertModel.from_pretrained('bert-base-uncased',output_hidden_states=True,output_attentions=False)  
        self.capsule = Caps_Layer(input_dim_capsule,num_capsule, dim_capsule,routings,T_epsilon,batch_size, kernel_size)                 
        self.dense_layer = Dense_Layer(dropout_p,num_capsule,dim_capsule)

    def forward(self, seq , attn_masks):
        '''
        Inputs:
            -seq : Tensor of shape [B, T] containing token ids of sequences
            -attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
        '''
        #Feeding the input to BERT model to obtain contextualized representations
              
        last_hidden_states, _, all_hidden_states = self.bert(seq, attention_mask=attn_masks)        
        caps_op = self.capsule(last_hidden_states) 
        output = self.dense_layer(caps_op)

        print('output shape ',output.shape)
        #Feeding cls_rep to the regression layer
        return output
    
    
class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y))