import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import torchvision.transforms as transforms
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm
from loguru import logger
import torch.nn.functional as F
from model_loader import load_model
from evaluate import mean_average_precision
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from torch.nn import Parameter
from losses import *

class BaseClassificationLoss(nn.Module):
    def __init__(self):
        super(BaseClassificationLoss, self).__init__()
        self.losses = {}

    def forward(self, logits, code_logits, labels, onehot=True):
        raise NotImplementedError

def get_imbalance_mask(sigmoid_logits, labels, nclass, threshold=0.7, imbalance_scale=-1):
    if imbalance_scale == -1:
        imbalance_scale = 1 / nclass

    mask = torch.ones_like(sigmoid_logits) * imbalance_scale

    # wan to activate the output
    mask[labels == 1] = 1

    # if predicted wrong, and not the same as labels, minimize it
    correct = (sigmoid_logits >= threshold) == (labels == 1)
    mask[~correct] = 1

    multiclass_acc = correct.float().mean()

    # the rest maintain "imbalance_scale"
    return mask, multiclass_acc


class warmup(BaseClassificationLoss):
    def __init__(self,
                 ce=1,
                 s=8,
                 m=0.2,
                 m_type='cos',
                 multiclass=False,
                 quan=0,
                 quan_type='cs',
                 multiclass_loss='label_smoothing',
                 **kwargs):
        super(warmup, self).__init__()
        self.ce = ce
        self.s = s
        self.m = m
        self.m_type = m_type
        self.multiclass = multiclass
        self.quan = quan
        self.quan_type = quan_type
        self.multiclass_loss = multiclass_loss
        assert multiclass_loss in ['bce', 'imbalance', 'label_smoothing']

    def compute_margin_logits(self, logits, labels):
        if self.m_type == 'cos':
            if self.multiclass:
                y_onehot = labels * self.m
                margin_logits = self.s * (logits - y_onehot)
            else:
                y_onehot = torch.zeros_like(logits)
                y_onehot.scatter_(1, torch.unsqueeze(labels, dim=-1), self.m)
                margin_logits = self.s * (logits - y_onehot)
        else:
            if self.multiclass:
                y_onehot = labels * self.m
                arc_logits = torch.acos(logits.clamp(-0.99999, 0.99999))
                logits = torch.cos(arc_logits + y_onehot)
                margin_logits = self.s * logits
            else:
                y_onehot = torch.zeros_like(logits)
                y_onehot.scatter_(1, torch.unsqueeze(labels, dim=-1), self.m)
                arc_logits = torch.acos(logits.clamp(-0.99999, 0.99999))
                logits = torch.cos(arc_logits + y_onehot)
                margin_logits = self.s * logits

        return margin_logits

    def forward(self, logits, code_logits, labels, onehot=True):
        if self.multiclass:
            if not onehot:
                labels = F.one_hot(labels, logits.size(1))
            labels = labels.float()
            margin_logits = self.compute_margin_logits(logits, labels)
            if self.multiclass_loss in ['bce', 'imbalance']:
                loss_ce = F.binary_cross_entropy_with_logits(margin_logits, labels, reduction='none')
                loss_ce_batch = F.cross_entropy(margin_logits, labels, reduction='none')
                if self.multiclass_loss == 'imbalance':
                    imbalance_mask, multiclass_acc = get_imbalance_mask(torch.sigmoid(margin_logits), labels,
                                                                        labels.size(1))
                    loss_ce = loss_ce * imbalance_mask
                    loss_ce = loss_ce.sum() / (imbalance_mask.sum() + 1e-7)
                    self.losses['multiclass_acc'] = multiclass_acc
                else:
                    loss_ce = loss_ce.mean()
            elif self.multiclass_loss in ['label_smoothing']:
                log_logits = F.log_softmax(margin_logits, dim=1)
                labels_scaled = labels / labels.sum(dim=1, keepdim=True)
                loss_ce = - (labels_scaled * log_logits).sum(dim=1)
                loss_ce_batch = loss_ce
                loss_ce = loss_ce.mean()
            else:
                raise NotImplementedError(f'unknown method: {self.multiclass_loss}')
        else:
            if onehot:
                labels = labels.argmax(1)
            margin_logits = self.compute_margin_logits(logits, labels)
            loss_ce = F.cross_entropy(margin_logits, labels)
            loss_ce_batch = F.cross_entropy(margin_logits, labels, reduction='none')


        if self.quan != 0:
            if self.quan_type == 'cs':
                quantization = (1. - F.cosine_similarity(code_logits, code_logits.detach().sign(), dim=1))
            elif self.quan_type == 'l1':
                quantization = torch.abs(code_logits - code_logits.detach().sign())
            else:  # l2
                quantization = torch.pow(code_logits - code_logits.detach().sign(), 2)
            quantization_batch = quantization
            quantization = quantization.mean()
        else:
            quantization = torch.tensor(0.).to(code_logits.device)
            quantization_batch = torch.zeros_like(loss_ce_batch)

        self.losses['ce'] = loss_ce
        self.losses['quan'] = quantization
        loss = self.ce * loss_ce + self.quan * quantization
        loss_batch = self.ce * loss_ce_batch + self.quan * quantization_batch
        return loss,loss_batch

def early(loss_ ,net, data, label, optimizer, criterion, nonzero_ratio, clip):
    net.train()
    pred,v,_ = net(data)
    if label.shape[1] > 2:
        label = label.argmax(1)
    
    loss = criterion(pred, label)
    loss += 0.1 * loss_
    loss.backward(retain_graph=True)
    
    to_concat_g = []
    to_concat_v = []
    for name, param in net.named_parameters():
        if param.dim() in [2, 4]:
            to_concat_g.append(param.grad.data.view(-1))
            to_concat_v.append(param.data.view(-1))
    all_g = torch.cat(to_concat_g)
    all_v = torch.cat(to_concat_v)
    metric = torch.abs(all_g * all_v)
    num_params = all_v.size(0)
    nz = int(nonzero_ratio * num_params)
    top_values, _ = torch.topk(metric, nz)
    thresh = top_values[-1]
    for name, param in net.named_parameters():
        if param.dim() in [2, 4]:
            mask = (torch.abs(param.data * param.grad.data) >= thresh).type(torch.cuda.FloatTensor)
            mask = mask * clip
            param.grad.data = mask * param.grad.data
    optimizer.step()
    optimizer.zero_grad()
    return loss

def train(train_dataloader,
          query_dataloader,
          retrieval_dataloader,
          multi_labels,
          code_length,
          num_features,
          alpha,
          beta,
          max_iter,
          arch,
          lr,
          device,
          verbose,
          topk,
          num_class,
          threshold,
          eta,
          temperature,
          evaluate_interval,
          tag,
          ):
    """
    Training model.

    Args
        train_dataloader(torch.evaluate.data.DataLoader): Training data loader.
        query_dataloader(torch.evaluate.data.DataLoader): Query data loader.
        retrieval_dataloader(torch.evaluate.data.DataLoader): Retrieval data loader.
        multi_labels(bool): True, if dataset is multi-labels.
        code_length(int): Hash code length.
        num_features(int): Number of features.
        alpha, beta(float): Hyper-parameters.
        max_iter(int): Number of iterations.
        arch(str): Model name.
        lr(float): Learning rate.
        device(torch.device): GPU or CPU.
        verbose(bool): Print log.
        evaluate_interval(int): Interval of evaluation.
        snapshot_interval(int): Interval of snapshot.
        topk(int): Calculate top k data points map.
        checkpoint(str, optional): Paht of checkpoint.
    Returns
        None
    """
    model = load_model(arch, code_length,num_class)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
    criterion_new = warm_up(multiclass=multi_labels)
    criterion = nn.CrossEntropyLoss()
    partion = init_partion_loss(
        multicrop = 0,
        tau = 0.1,
        T = 0.25,
        me_max=True)
    model.train()
    names = locals()
    warm_up = 30
    start0 = time.time()
    for epoch in range(max_iter):
        start = time.time()
        noiseLevel = 0.6
        num_gradual = 10
        clip_narry = np.linspace(1-noiseLevel, 1, num=num_gradual)
        clip_narry = clip_narry[::-1]
        if epoch < num_gradual:
            clip = clip_narry[epoch]
        clip = (1 - noiseLevel)
        names['lists' + str(epoch) ] = []
        n_batch = len(train_dataloader)
        for iters, (data, data_aug, target, index) in enumerate(train_dataloader):
            data = data.to(device)
            batch_size = data.shape[0]
            data_aug = data_aug.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            logit, v,_= model(data)
            logit_aug, v_aug,_= model(data)

            if epoch < warm_up:
                if target.shape[1] > 2:
                    target = target.argmax(1)
                loss = criterion(logit, target)
                loss.backward()
                optimizer.step()
            else:
                loss, loss_batch = criterion_new(logit, v, target)
                loss_aug, loss_aug_batch = criterion_new(logit_aug,v_aug,target)
                difference_batch = torch.abs(loss_batch-loss_aug_batch)
                cal_unclean = 0
                cal_clean = 0
                clean_threshold_, indice = torch.sort(difference_batch, descending=True)
                clean_threshold = clean_threshold_[int(0.6*(difference_batch.shape[0]))]
                # Noisy Sample Selection
                for i,val in enumerate(difference_batch):    
                    if val >= clean_threshold:
                        if cal_unclean == 0:
                            unclean_data = data[i].unsqueeze(0).clone()
                            unclean_aug_data = data_aug[i].unsqueeze(0).clone()
                            unclean_v = v[i].unsqueeze(0).clone()
                            unclean_aug_v = v_aug[i].unsqueeze(0).clone()
                        else:
                            unclean_data = torch.cat((unclean_data,data[i].unsqueeze(0).clone()),0)
                            unclean_aug_data = torch.cat((unclean_aug_data,data_aug[i].unsqueeze(0).clone()),0)
                            unclean_v = torch.cat((unclean_v,v[i].unsqueeze(0).clone()),0)
                            unclean_aug_v = torch.cat((unclean_aug_v,v_aug[i].unsqueeze(0).clone()),0)
                        cal_unclean += 1
                    else:
                        if cal_clean == 0:
                            clean_data = data[i].unsqueeze(0).clone()
                            clean_aug_data = data_aug[i].unsqueeze(0).clone()
                            clean_v = v[i].unsqueeze(0).clone()
                            clean_aug_v = v_aug[i].unsqueeze(0).clone()
                            target_clean = target[i].unsqueeze(0).clone()
                        else:
                            clean_data = torch.cat((clean_data,data[i].unsqueeze(0).clone()),0)
                            clean_aug_data = torch.cat((clean_aug_data,data_aug[i].unsqueeze(0).clone()),0)
                            clean_v = torch.cat((clean_v,v[i].unsqueeze(0).clone()),0)
                            clean_aug_v = torch.cat((clean_aug_v,v_aug[i].unsqueeze(0).clone()),0)
                            target_clean = torch.cat((target_clean,target[i].unsqueeze(0).clone()),0)
                            
                        cal_clean += 1
                (ploss, me_max) = partion(
                            anchor_views=unclean_v,
                            anchor_supports=clean_v,
                            anchor_support_labels=target_clean,
                            target_views=unclean_aug_v,
                            target_supports=clean_aug_v,
                            target_support_labels=target_clean)
                loss = ploss + me_max
                loss = early(loss, model, clean_data, target_clean, optimizer, criterion, clip, clip)

        end = time.time() - start
        print('[Epoch:{}/{}][loss:{:.4f}][time:{}]'.format(epoch+1, max_iter, loss.item(),end))
        if (epoch % evaluate_interval == evaluate_interval-1):
            end_all = time.time() - start0
            mAP = evaluate(model,
                            query_dataloader,
                            retrieval_dataloader,
                            code_length,
                            device,
                            topk,
                            multi_labels,
                            )
            print('[iter:{}/{}][map:{:.4f}][time:{}]'.format(
                epoch+1,
                max_iter,
                mAP,
                end_all,
            ))

    mAP = evaluate(model,
                   query_dataloader,
                   retrieval_dataloader,
                   code_length,
                   device,
                   topk,
                   multi_labels,
                   )
    torch.save({'iteration': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join('checkpoints', 'resume_{}.t'.format(code_length)))
    logger.info('Training finish, [iteration:{}][map:{:.4f}]'.format(epoch+1, mAP))


def evaluate(model, query_dataloader, retrieval_dataloader, code_length, device, topk, multi_labels):
    model.eval()
    # Generate hash code
    query_code = generate_code(model, query_dataloader, code_length, device)
    retrieval_code = generate_code(model, retrieval_dataloader, code_length, device)
    # One-hot encode targets
    if multi_labels:
        onehot_query_targets = query_dataloader.dataset.get_targets().to(device)
        onehot_retrieval_targets = retrieval_dataloader.dataset.get_targets().to(device)
    else:
        onehot_query_targets = query_dataloader.dataset.get_onehot_targets().to(device)
        onehot_retrieval_targets = retrieval_dataloader.dataset.get_onehot_targets().to(device)
    mAP = mean_average_precision(
        query_code,
        retrieval_code,
        onehot_query_targets,
        onehot_retrieval_targets,
        device,
        topk,
    )
    model.train()
    return mAP


def generate_code(model, dataloader, code_length, device):
    """
    Generate hash code.

    Args
        model(torch.nn.Module): CNN model.
        dataloader(torch.evaluate.data.DataLoader): Data loader.
        code_length(int): Hash code length.
        device(torch.device): GPU or CPU.

    Returns
        code(torch.Tensor): Hash code.
    """
    with torch.no_grad():
        N = len(dataloader.dataset)
        code = torch.zeros([N, code_length])
        for data, _, index in dataloader:
            data = data.to(device)
            _,outputs,_= model(data)
            code[index, :] = outputs.sign().cpu()

    return code


def generate_similarity_weight_matrix(features, alpha, beta, threshold, k_positive, k_negative, Classes):
    """
    Generate similarity and confidence matrix.

    Args
        features(torch.Tensor): Features.
        alpha, beta(float): Hyper-parameters.

    Returns
        S(torch.Tensor): Similarity matrix.
    """
    # # Cosine similarity
    cos_dist = squareform(pdist(features.numpy(), 'cosine'))
    features = features.numpy()

    # Construct similarity matrix
    S = (cos_dist <= threshold) * 1.0 + (cos_dist > threshold ) * -1.0
    max_cnt, max_cos = 0, 0
    interval = 1. / 100
    cur = 0
    for i in range(100):
        cur_cnt = np.sum((cos_dist > cur) & (cos_dist < cur + interval))
        if max_cnt < cur_cnt:
            max_cnt = cur_cnt
            max_cos = cur
        cur += interval



    # Split features into two parts
    flat_cos_dist = cos_dist.reshape((-1, 1))
    left = flat_cos_dist[np.where(flat_cos_dist <= max_cos)[0]]
    right = flat_cos_dist[np.where(flat_cos_dist > max_cos)[0]]

    # Reconstruct gaussian distribution
    left = np.concatenate([left, 2 * max_cos - left])
    right = np.concatenate([2 * max_cos - right, right])

    # Model data using gaussian distribution
    left_mean, left_std = norm.fit(left)
    right_mean, right_std = norm.fit(right)
    dist_up = right_mean + beta * right_std
    dist_down = left_mean - alpha * left_std

    def weight_norm(x, threshold):
        weight_f = (x>threshold)* (norm.cdf((x-right_mean)/right_std)-norm.cdf((threshold-right_mean)/right_std))/(1-norm.cdf((threshold-right_mean)/right_std)) + \
         (x<=threshold) * (norm.cdf((threshold-left_mean)/left_std)-norm.cdf((x-left_mean)/left_std) )/ (norm.cdf((threshold-left_mean)/left_std))
        return weight_f
    
    weight_1 = np.clip(weight_norm(cos_dist, threshold), 0, 1)


    # weight according to clustering
    features_norm = (features.T/ np.linalg.norm(features,axis=1)).T
    sp_cluster = SpectralClustering(n_clusters=Classes, random_state=0, assign_labels="discretize").fit(features_norm)
    A = sp_cluster.labels_[np.newaxis, :] #label vector
    # kmeans = KMeans(n_clusters=Classes, random_state=0, init='k-means++').fit(features_norm)
    # A = kmeans.labels_[np.newaxis, :] #label vector
    weight_2 =  ((((A - A.T) == 0)-1/2)*2* S +1)/2
    W = weight_1 * weight_2
    return torch.FloatTensor(S), torch.FloatTensor(W)


def extract_features(model, dataloader, num_features, device, verbose):
    """
    Extract features.
    """
    model.eval()
    model.set_extract_features(True)
    features_1 = torch.zeros(dataloader.dataset.data.shape[0], num_features)
    features_2 = torch.zeros(dataloader.dataset.data.shape[0], num_features)
    with torch.no_grad():
        N = len(dataloader)
        for i, (data_1, data_2 ,_, index) in enumerate(dataloader):
            if verbose:
                logger.debug('[Batch:{}/{}]'.format(i+1, N))
            data_1 = data_1.to(device)
            data_2 = data_2.to(device)
            features_1[index, :] = model(data_1).cpu()
            features_2[index, :] = model(data_2).cpu()

    model.set_extract_features(False)
    model.train()
    return features_1, features_2

class SEM_CON_Loss(nn.Module):
    def __init__(self):
        super(SEM_CON_Loss, self).__init__()

    def forward(self, H, W, S):
        loss = (W * S.abs() * (H - S).pow(2)).sum() / (H.shape[0] ** 2)
        return loss





    


    
