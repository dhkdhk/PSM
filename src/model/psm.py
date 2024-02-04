import math

import torch
import torch.nn as nn
import torch
from .memory_bank import MemoryBankModule
import model.network as models
import torch.nn.functional as F
torch.set_printoptions(threshold=2000)



class PSM(nn.Module):
    """
    Build a SNCLR model with a base encoder, a momentum encoder, and two MLPs
    """

    def __init__(self, args, dim=256, mlp_dim=4096, T=1.0, hidden_dim=384,
                 bank_size=98304, model='res', world_size=8, threshold=False, topk=5,
                 batch=256):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super(PSM, self).__init__()
        self.args=args
        self.m=args.queue_momentum
        self.T = T
        self.topk = topk
        self.batch = batch
        self.threshold = threshold
        self.threshold = True
        self.world_size = world_size
        self.encoder_q = getattr(models, args.model)(
            args, num_classes=128)  # Query Encoder
        self.encoder_k = getattr(models, args.model)(
            args, num_classes=128)  # Key Encoder
        self.scale = dim ** -0.5

        self.a=args.a
        self.b=args.b
        self.reload_frequency = args.reload_frequency
        self.epoch = args.n_epochs
        self.swap_param = 2 / 3
        self.aa=args.aa

        self.base_encoder = self.encoder_q
        self.momentum_encoder = self.encoder_k
        # self.predictor1 = MLPHead(in_channels=128)
        self._build_projector_and_predictor_mlps(dim, mlp_dim)


        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)
            param_m.requires_grad = False

        self.memory_bank = NNMemoryBankModuleNormMultiKeep(size=bank_size, topk=self.topk)

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.fc.weight.shape[1]
        print(f'dim {hidden_dim}-{mlp_dim}-{dim}')
        del self.base_encoder.fc, self.momentum_encoder.fc

        # projectors
        # self.base_encoder.fc= self._build_mlp(3, hidden_dim, mlp_dim, dim)
        # self.momentum_encoder.fc = self._build_mlp(3, hidden_dim, mlp_dim, dim)
        self.base_encoder.fc= self._build_mlp(2, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)
        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim)
        # self.predictor2 = self._build_mlp(2, dim, mlp_dim, dim)


    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def att_clc(self, q_pre, k_pre,epoch):
        q = q_pre
        k = k_pre

        att_mask = self.generation_mask(self.topk, self.batch).cuda()
        attn = q @ k.transpose(-2, -1)

        attn = attn + att_mask

        attn = nn.Softmax(dim=-1)(attn)

        return attn


    def generation_mask(self, topk, batch):
        mask_ = torch.eye(batch)
        mask_ = (mask_ - 1) * 999
        mask = mask_.repeat(topk,1).reshape(topk,batch,-1).permute(2,1,0).reshape(batch,topk*batch)

        return mask

    def soft_constrastive(self, q, k, k_keep, q_pre,pos,epoch):

        device=q.device
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        k_keep = nn.functional.normalize(k_keep, dim=1)
        q_pre = nn.functional.normalize(q_pre, dim=1)
        pos=nn.functional.normalize(pos, dim=1)


        logits1 = torch.einsum('nc,mc->nm', [q, k_keep]) # 128， 20480 gor 32 gpus   128, 10240  128,


        a = logits1.shape[0]
        b = logits1.shape[1]
        pos = torch.einsum(
                'b d, b d -> b', q, pos)[:, None]



        pos_mask=select_pos_mask(logits1,self.topk+1).to(device)             # 256*(1+5)

        pospro = logits1.masked_select(pos_mask).view(a, -1)
        neg_mask=select_neg_mask(logits1,self.topk+1).to(device)               # 256*1530
        neg = logits1.masked_select(neg_mask).view(a, -1)


        size=neg.shape[1]
        logits_pos1 = pos.repeat(1, size)
        x1 = neg - logits_pos1 - self.b
        x2 = torch.pow(x1, 2)
        x3 = torch.mul(x2, -self.a)
        x4 = torch.exp(x3)
        neg_sampler = torch.distributions.bernoulli.Bernoulli(x4.clamp(0.0, 0.999))
        selected_mask = neg_sampler.sample()
        x5 = neg * selected_mask

        x55 = torch.exp(x5 /self.T)
        value_to_remove = 1
        mask = torch.ones_like(x55)
        mask[x55 == value_to_remove] = 0
        x55 = x55 * mask
        Ng = x55.sum(dim=-1)
        z=torch.exp(neg / self.T)

        labels = self.att_clc(q_pre, k,epoch)         # attention module
        b, all_b = labels.shape


        one_label = torch.eye(b).cuda()
        fina_label = torch.cat((one_label.reshape(b,-1,b).unsqueeze(dim=-1), labels.reshape(b, -1, b, self.topk)), dim = -1).reshape(b, -1)

        labelsnew_mask = select_pos_mask(fina_label, self.topk + 1).to(device)
        labelsnew = fina_label.masked_select(labelsnew_mask).view(a, -1)

        labelsnew_1 =labelsnew.clone()
        labelsnew_1[labelsnew_1 >0]=1


        pos1=torch.exp((pospro) / self.T)

        if self.threshold:
            pos1 = pos1 * labelsnew_1
            labelsnew=labelsnew_1


        pos1=pos1.sum(dim=-1)
        logit11=torch.zeros_like(pospro)

        for i in range(self.topk+1):
            pp=torch.exp(pospro[:,i] / self.T)
            logit11[:, i] = - torch.log(pp / (pos1 + Ng))


        loss=torch.mean(torch.sum(logit11 *labelsnew, dim=-1),dim=-1)
        return loss




    def forward(self, x1, x2,epoch):

        q1_pre = self.base_encoder(x1)
        q2_pre = self.base_encoder(x2)
        q1 = self.predictor(q1_pre)
        q2 = self.predictor(q2_pre)

        with torch.no_grad():  # no gradient
            self._update_momentum_encoder(self.m)  # update the momentum encoder

            # compute momentum features as targets
            k1_ = self.momentum_encoder(x1)
            k2_ = self.momentum_encoder(x2)

            k1, k1_keep,output1,bank1 = self.memory_bank(k1_, update=False)
            k2, k2_keep,output2,bank2= self.memory_bank(k2_, update=True)

        L_soft = self.soft_constrastive(q1, k2, k2_keep, q1_pre, output2,
                                                    epoch) + self.soft_constrastive(q2, k1, k1_keep, q2_pre,
                                                                                              output1, epoch)
        L_hard =hard_contrastive(q1,k2_,self.args)+hard_contrastive(q2,k1_,self.args)

        loss =  (L_soft/2)+self.aa*(L_hard/2)

        return loss ,self.topk



def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask

def hard_contrastive(out_1, out_2, args):
    feat_q = F.normalize(out_1, dim=1)
    feat_k = F.normalize(out_2, dim=1)
    device=feat_q.device

    logits_pos = torch.einsum(
        'b d, b d -> b', out_1, out_2)[:, None]  #

    logits_pos_true = torch.einsum(
        'b d, b d -> b', feat_q, feat_k)[:, None]
    pos = torch.exp(logits_pos_true / args.temperature).to(device)
    pos = torch.cat([pos, pos], dim=0)

    out = torch.cat([out_1, out_2], dim=0)
    logits_neg = torch.einsum(
        'b d, d k -> b k', out, out.t().contiguous())
    mask = get_negative_mask(args.batch_size).to(device)
    neg = logits_neg.masked_select(mask).view(2 * args.batch_size, -1)


    logits_pos1 = torch.cat([logits_pos, logits_pos], dim=0)
    logits_pos2 = logits_pos1.repeat(1, 2 * 256 - 2)
    x1 = neg - logits_pos2 - args.b
    x2 = torch.pow(x1, 2)
    x3 = torch.mul(x2, -args.a)
    x4 = torch.exp(x3)
    neg_sampler = torch.distributions.bernoulli.Bernoulli(x4.clamp(0.0, 0.999))
    selected_mask = neg_sampler.sample()


    out = torch.cat([feat_q, feat_k], dim=0)
    logits_neg_true = torch.einsum(
        'b d, d k -> b k', out, out.t().contiguous())
    mask = get_negative_mask(args.batch_size).to(device)
    neg_true = logits_neg_true.masked_select(mask).view(2 * args.batch_size, -1)

    x5 = neg_true * selected_mask.to(device)
    x55 = torch.exp(x5 / args.temperature).to(device)

    value_to_remove = 1
    mask = torch.ones_like(x55)
    mask[x55 == value_to_remove] = 0
    x55 = x55 * mask

    # Ng = torch.sum(x55).to(device)
    Ng = x55.sum(dim=-1)

    loss = (- torch.log(pos / (pos + Ng))).mean()
    return loss


@torch.no_grad()
def concat_all_gather(tensor):
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class NNMemoryBankModuleNormMultiKeep(MemoryBankModule):
    """Nearest Neighbour Memory Bank implementation

    This class implements a nearest neighbour memory bank as described in the
    NNCLR paper[0]. During the forward pass we return the nearest neighbour
    from the memory bank.

    [0] NNCLR, 2021, https://arxiv.org/abs/2104.14548

    Attributes:
        size:
            Number of keys the memory bank can store. If set to 0,
            memory bank is not used.

    Examples:
        >>> model = NNCLR(backbone)
        >>> criterion = NTXentLoss(temperature=0.1)
        >>>
        >>> nn_replacer = NNmemoryBankModule(size=2 ** 16)
        >>>
        >>> # forward pass
        >>> (z0, p0), (z1, p1) = model(x0, x1)
        >>> z0 = nn_replacer(z0.detach(), update=False)
        >>> z1 = nn_replacer(z1.detach(), update=True)
        >>>
        >>> loss = 0.5 * (criterion(z0, p1) + criterion(z1, p0))
    """
    def __init__(self, size: int = 2 ** 16, topk=5):                         # 2**16设置的是bank大小
        super(NNMemoryBankModuleNormMultiKeep, self).__init__(size)
        self.topk = topk
    def set_topk(self, new_topk):
        self.topk = new_topk
    def forward(self,
                output: torch.Tensor,
                update: bool = False):
        """Returns nearest neighbour of output tensor from memory bank

        Args:
            output: The torch tensor for which you want the nearest neighbour
            update: If `True` updated the memory bank by adding output to it

        """

        output, bank = \
            super(NNMemoryBankModuleNormMultiKeep, self).forward(output, update=update)
        bank = bank.to(output.device).t()

        output_normed = torch.nn.functional.normalize(output, dim=1)
        bank_normed = torch.nn.functional.normalize(bank, dim=1)

        similarity_matrix = \
            torch.einsum("nd,md->nm", output_normed, bank_normed)

        index_nearest_neighbours = torch.topk(similarity_matrix, self.topk)[1].reshape(-1)
        # 256*5=1280， 256个每个都找最大的5个，一共1280个


        nearest_neighbours = torch.index_select(bank, dim=0, index=index_nearest_neighbours)
        b, dim = output.shape

        res = torch.cat((output.unsqueeze(dim=1), nearest_neighbours.reshape(b, self.topk, dim)), dim=1).reshape(-1, dim)
        # 原本256个+新选出来的1280个=1536*256

        return nearest_neighbours, res ,output,bank_normed
        # 1280*256     1536*256        注意这里的nearest_neighbours不是直接cat到后面的，是紧跟着每一个正样本
        """
        pos 
        near*5
        pos2
        near*5        所以后面的label要那样计算
        """

def select_neg_mask(logit,k):
    device = logit.device
    # 初始化一个列表，用于存储每行对应区间的元素
    a=logit.shape[0]
    b=logit.shape[1]
    negative_mask = torch.ones((a,b), dtype=bool)

    selected_rows = []
    # 逐行获取每行的对应区间元素，并存储在selected_rows列表中
    for i in range(logit.shape[0]):  # 遍历每一行
        negative_mask[i, i*k:i*k+k]=0  # 获取对应区间的元素

    # neg = logit.masked_select(negative_mask).view(a, -1)


    return  negative_mask

def select_pos_mask(logit,k):
    device=logit.device
    a=logit.shape[0]
    b=logit.shape[1]
    negative_mask = torch.zeros((a,b), dtype=bool)

    selected_rows = []
    # 逐行获取每行的对应区间元素，并存储在selected_rows列表中
    for i in range(logit.shape[0]):  # 遍历每一行
        negative_mask[i, i*k:i*k+k]=1  # 获取对应区间的元素

    # pos = logit.masked_select(negative_mask).view(a, -1)
    return  negative_mask