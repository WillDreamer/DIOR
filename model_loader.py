import torch
import torch.nn as nn
import torchvision.models as models
import os
from torch.nn import Parameter

class CosSim(nn.Module):
    def __init__(self, nfeat, nclass, learn_cent=True):
        super(CosSim, self).__init__()
        self.nfeat = nfeat
        self.nclass = nclass
        self.learn_cent = learn_cent

         # if no centroids, by default just usual weight
        codebook = torch.randn(nclass, nfeat)

        self.centroids = nn.Parameter(codebook.clone())
        if not learn_cent:
            self.centroids.requires_grad_(True)

    def forward(self, x):
        norms = torch.norm(x, p=2, dim=-1, keepdim=True)
        nfeat = torch.div(x, norms)

        norms_c = torch.norm(self.centroids, p=2, dim=-1, keepdim=True)
        ncenters = torch.div(self.centroids, norms_c)
        logits = torch.matmul(nfeat, torch.transpose(ncenters, 0, 1))

        return logits

def load_model(arch, code_length, num_cluster=30):
    """
    Load CNN model.

    Args
        arch(str): Model name.
        code_length(int): Hash code length.

    Returns
        model(torch.nn.Module): CNN model.
    """
    if arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        model.classifier = model.classifier[:-2]
        model = ModelWrapper(model, 4096, code_length,num_cluster)
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        # relu_list = [1,3,6,8,11,13,15,18,20,22,25,27,29]
        # for i in relu_list:
        #     model.features[i] = nn.ReLU(inplace=False)
        model.classifier = model.classifier[:-3]
        # model.classifier[1] = nn.ReLU(inplace=False)

        model = ModelWrapper(model, 4096, code_length,num_cluster)
    else:
        raise ValueError("Invalid model name!")

    return model


class ModelWrapper(nn.Module):
    """
    Add tanh activate function into model.

    Args
        model(torch.nn.Module): CNN model.
        last_node(int): Last layer outputs size.
        code_length(int): Hash code length.
    """
    def __init__(self, model, last_node, code_length, num_cluster):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.code_length = code_length
        self.hash_layer = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(last_node, code_length),
            nn.Tanh(),
        )
        # self.cluster_layer = Parameter(torch.Tensor(num_cluster, code_length))
        # self.head = nn.Sequential(nn.Linear(last_node, num_cluster),
        #     nn.Softmax(dim=1))
        # torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # Extract features
        self.ce_fc = CosSim(code_length, num_cluster, learn_cent=False)


    def forward(self, x):
        
        feature = self.model(x)
        y = self.hash_layer(feature)
        logit = self.ce_fc(y)
        return logit, y, feature

    def set_extract_features(self, flag):
        """
        Extract features.

        Args
            flag(bool): true, if one needs extract features.
        """
        self.extract_features = flag


