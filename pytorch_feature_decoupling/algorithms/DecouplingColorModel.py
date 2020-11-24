import pdb

import os
import time
import numpy as np

import torch

from . import Algorithm

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class DecouplingColorModel(Algorithm):
    def __init__(self, opt):
        self.lambda_loss = opt['lambda_loss']
        self.gama = opt['gama']
        self.permutations = self.generatePermutationOrders()
        Algorithm.__init__(self, opt)

    def generatePermutationOrders(self):
        result = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    if i != j and i != k and j != k:
                        result.append([i, j, k])
        return result

    def allocate_tensors(self):
        self.tensors = {}
        self.tensors['dataX'] = torch.FloatTensor()
        self.tensors['index'] = torch.LongTensor()
        self.tensors['index_index'] = torch.LongTensor()
        self.tensors['labels'] = torch.LongTensor()
        
    def train_step(self, batch):
        start = time.time()
        #*************** LOAD BATCH (AND MOVE IT TO GPU) ********
        self.tensors['dataX'].resize_(batch[0].size()).copy_(batch[0])
        dataX_0 = torch.tensor(self.tensors['dataX'])
        dataX_90 = torch.flip(torch.transpose(self.tensors['dataX'],2,3),[2])
        dataX_180 = torch.flip(torch.flip(self.tensors['dataX'],[2]),[3])
        dataX_270 = torch.transpose(torch.flip(self.tensors['dataX'],[2]),2,3)

        dataXPermutated0 = dataX_0[:, self.permutations[0], :, :]
        dataXPermutated1 = dataX_0[:, self.permutations[1], :, :]
        dataXPermutated2 = dataX_0[:, self.permutations[2], :, :]
        dataXPermutated3 = dataX_0[:, self.permutations[3], :, :]
        dataXPermutated4 = dataX_0[:, self.permutations[4], :, :]
        dataXPermutated5 = dataX_0[:, self.permutations[5], :, :]

        dataXPermutated6 = dataX_90[:, self.permutations[0], :, :]
        dataXPermutated7 = dataX_90[:, self.permutations[1], :, :]
        dataXPermutated8 = dataX_90[:, self.permutations[2], :, :]
        dataXPermutated9 = dataX_90[:, self.permutations[3], :, :]
        dataXPermutated10 = dataX_90[:, self.permutations[4], :, :]
        dataXPermutated11 = dataX_90[:, self.permutations[5], :, :]

        dataXPermutated12 = dataX_180[:, self.permutations[0], :, :]
        dataXPermutated13 = dataX_180[:, self.permutations[1], :, :]
        dataXPermutated14 = dataX_180[:, self.permutations[2], :, :]
        dataXPermutated15 = dataX_180[:, self.permutations[3], :, :]
        dataXPermutated16 = dataX_180[:, self.permutations[4], :, :]
        dataXPermutated17 = dataX_180[:, self.permutations[5], :, :]
        
        dataXPermutated18 = dataX_270[:, self.permutations[0], :, :]
        dataXPermutated19 = dataX_270[:, self.permutations[1], :, :]
        dataXPermutated20 = dataX_270[:, self.permutations[2], :, :]
        dataXPermutated21 = dataX_270[:, self.permutations[3], :, :]
        dataXPermutated22 = dataX_270[:, self.permutations[4], :, :]
        dataXPermutated23 = dataX_270[:, self.permutations[5], :, :]

        dataX = torch.stack([dataXPermutated0, dataXPermutated1, dataXPermutated2, dataXPermutated3, dataXPermutated4, dataXPermutated5, \
                            dataXPermutated6, dataXPermutated7, dataXPermutated8, dataXPermutated9, dataXPermutated10, dataXPermutated11, \
                            dataXPermutated12, dataXPermutated13, dataXPermutated14, dataXPermutated15, dataXPermutated16, dataXPermutated17, \
                            dataXPermutated18, dataXPermutated19, dataXPermutated20, dataXPermutated21, dataXPermutated22, dataXPermutated23 \
                                ], dim=1)
        batch_size, permutations, channels, height, width = dataX.size()
        dataX = dataX.view([batch_size*permutations, channels, height, width])
        
        self.tensors['labels'].resize_(batch[1].size()).copy_(batch[1])
        self.tensors['index'].resize_(batch[2].size()).copy_(batch[2])
        labels = self.tensors['labels']
        index = self.tensors['index']
        #********************************************************
        batch_load_time = time.time() - start

        start = time.time()
        #************ FORWARD THROUGH NET ***********************
        for _, network in self.networks.items():
            for param in network.parameters():
                param.requires_grad = True

        with torch.set_grad_enabled(True):
            feature = self.networks['feature'](dataX)
            feature_rot, feature_invariance = torch.split(feature, 2048, dim=1)

            pred = self.networks['classifier'](feature_rot)

            feature_invariance_instance = feature_invariance[0::24,:] + feature_invariance[1::24,:] + feature_invariance[2::24,:] + feature_invariance[3::24,:] \
                                        + feature_invariance[4::24,:] + feature_invariance[5::24,:] + feature_invariance[6::24,:] + feature_invariance[7::24,:] \
                                        + feature_invariance[8::24,:] + feature_invariance[9::24,:] + feature_invariance[10::24,:] + feature_invariance[11::24,:] \
                                        + feature_invariance[12::24,:] + feature_invariance[13::24,:] + feature_invariance[14::24,:] + feature_invariance[15::24,:] \
                                        + feature_invariance[16::24,:] + feature_invariance[17::24,:] + feature_invariance[18::24,:] + feature_invariance[19::24,:] \
                                        + feature_invariance[20::24,:] + feature_invariance[21::24,:] + feature_invariance[22::24,:] + feature_invariance[23::24,:]
            feature_invariance_instance = torch.mul(feature_invariance_instance, 1.0/24.0)
            feature_nce_norm = self.networks['norm'](feature_invariance_instance)
        
        with torch.set_grad_enabled(False):
            self.tensors['index_index'].resize_(torch.Size([int(index.size(0)/24)])).copy_(index[0::24])
            index_instance = self.tensors['index_index']
            feature_invariance_instance_mean = torch.unsqueeze(feature_invariance_instance,1).expand(-1,24,-1).clone()
            feature_invariance_instance_mean = feature_invariance_instance_mean.view(24*len(feature_invariance_instance),2048)
        #********************************************************

        #*************** COMPUTE LOSSES *************************
        #weight = torch.tensor(np.array(self.train_weight)[idx_train], dtype=torch.float, device=labels.device, requires_grad=False)
        with torch.set_grad_enabled(True):
            loss_cls_each = self.criterions['loss_cls'](pred, labels)
            loss_cls = torch.sum(loss_cls_each)/loss_cls_each.shape[0]

            loss_mse = self.criterions['loss_mse'](feature_invariance, feature_invariance_instance_mean)

            output_nce = self.criterions['nce_average'](feature_nce_norm, index_instance)
            loss_nce = self.criterions['nce_criterion'](output_nce, index_instance)

            loss_total = self.lambda_loss['cls']*loss_cls + self.lambda_loss['mse']*loss_mse + self.lambda_loss['nce']*loss_nce

        record = {}
        record['prec_cls'] = accuracy(pred, labels, topk=(1,))[0].item()

        record['loss'] = loss_total.item()
        record['loss_cls'] = loss_cls.item()
        record['loss_mse'] = loss_mse.item()
        record['loss_nce'] = loss_nce.item()
        #********************************************************

        #****** BACKPROPAGATE AND APPLY OPTIMIZATION STEP *******
        self.optimizers['feature'].zero_grad()
        self.optimizers['classifier'].zero_grad()
        self.optimizers['norm'].zero_grad()
        loss_total.backward()
        self.optimizers['feature'].step()
        self.optimizers['classifier'].step()
        self.optimizers['norm'].step()
        #********************************************************
        batch_process_time = time.time() - start
        total_time = batch_process_time + batch_load_time
        record['load_time'] = 100*(batch_load_time/total_time)
        record['process_time'] = 100*(batch_process_time/total_time)

        return record

    def evaluation_step(self, batch):
        start = time.time()
        #*************** LOAD BATCH (AND MOVE IT TO GPU) ********
        self.tensors['dataX'].resize_(batch[0].size()).copy_(batch[0])
        dataX_0 = torch.tensor(self.tensors['dataX'])
        dataX_90 = torch.flip(torch.transpose(self.tensors['dataX'],2,3),[2])
        dataX_180 = torch.flip(torch.flip(self.tensors['dataX'],[2]),[3])
        dataX_270 = torch.transpose(torch.flip(self.tensors['dataX'],[2]),2,3)

        dataXPermutated0 = dataX_0[:, self.permutations[0], :, :]
        dataXPermutated1 = dataX_0[:, self.permutations[1], :, :]
        dataXPermutated2 = dataX_0[:, self.permutations[2], :, :]
        dataXPermutated3 = dataX_0[:, self.permutations[3], :, :]
        dataXPermutated4 = dataX_0[:, self.permutations[4], :, :]
        dataXPermutated5 = dataX_0[:, self.permutations[5], :, :]

        dataXPermutated6 = dataX_90[:, self.permutations[0], :, :]
        dataXPermutated7 = dataX_90[:, self.permutations[1], :, :]
        dataXPermutated8 = dataX_90[:, self.permutations[2], :, :]
        dataXPermutated9 = dataX_90[:, self.permutations[3], :, :]
        dataXPermutated10 = dataX_90[:, self.permutations[4], :, :]
        dataXPermutated11 = dataX_90[:, self.permutations[5], :, :]

        dataXPermutated12 = dataX_180[:, self.permutations[0], :, :]
        dataXPermutated13 = dataX_180[:, self.permutations[1], :, :]
        dataXPermutated14 = dataX_180[:, self.permutations[2], :, :]
        dataXPermutated15 = dataX_180[:, self.permutations[3], :, :]
        dataXPermutated16 = dataX_180[:, self.permutations[4], :, :]
        dataXPermutated17 = dataX_180[:, self.permutations[5], :, :]
        
        dataXPermutated18 = dataX_270[:, self.permutations[0], :, :]
        dataXPermutated19 = dataX_270[:, self.permutations[1], :, :]
        dataXPermutated20 = dataX_270[:, self.permutations[2], :, :]
        dataXPermutated21 = dataX_270[:, self.permutations[3], :, :]
        dataXPermutated22 = dataX_270[:, self.permutations[4], :, :]
        dataXPermutated23 = dataX_270[:, self.permutations[5], :, :]

        dataX = torch.stack([dataXPermutated0, dataXPermutated1, dataXPermutated2, dataXPermutated3, dataXPermutated4, dataXPermutated5, \
                            dataXPermutated6, dataXPermutated7, dataXPermutated8, dataXPermutated9, dataXPermutated10, dataXPermutated11, \
                            dataXPermutated12, dataXPermutated13, dataXPermutated14, dataXPermutated15, dataXPermutated16, dataXPermutated17, \
                            dataXPermutated18, dataXPermutated19, dataXPermutated20, dataXPermutated21, dataXPermutated22, dataXPermutated23 \
                                ], dim=1)
        batch_size, permutations, channels, height, width = dataX.size()
        dataX = dataX.view([batch_size*permutations, channels, height, width])

        self.tensors['labels'].resize_(batch[1].size()).copy_(batch[1])
        #********************************************************
        batch_load_time = time.time() - start

        start = time.time()
        #************ FORWARD THROUGH NET ***********************
        for _, network in self.networks.items():
            for param in network.parameters():
                param.requires_grad = False

        with torch.set_grad_enabled(False):
            feature = self.networks['feature'](dataX)
            feature_rot, feature_invariance = torch.split(feature, 2048, dim=1)
            pred_rot = self.networks['classifier'](feature_rot)
            pred_inv = self.networks['classifier'](feature_invariance)
        #********************************************************

        #*************** COMPUTE LOSSES *************************
        with torch.set_grad_enabled(False):
            loss_rot_each = self.criterions['loss_cls'](pred_rot, self.tensors['labels'])
            loss_inv_each = self.criterions['loss_cls'](pred_inv, self.tensors['labels'])
            loss_rot = torch.sum(loss_rot_each)/loss_rot_each.shape[0]
            loss_inv = torch.sum(loss_inv_each)/loss_inv_each.shape[0]   
        record = {}
        record['prec_rot'] = accuracy(pred_rot, self.tensors['labels'], topk=(1,))[0].item()
        record['prec_inv'] = accuracy(pred_inv, self.tensors['labels'], topk=(1,))[0].item()
        record['loss_rot'] = loss_rot.item()
        record['loss_inv'] = loss_inv.item()
        #********************************************************
        batch_process_time = time.time() - start
        total_time = batch_process_time + batch_load_time
        record['load_time'] = 100*(batch_load_time/total_time)
        record['process_time'] = 100*(batch_process_time/total_time)

        return record

    def getFeatures(self, batch):
        #*************** LOAD BATCH (AND MOVE IT TO GPU) ********
        self.tensors['dataX'].resize_(batch[0].size()).copy_(batch[0])

        #************ FORWARD THROUGH NET ***********************
        for _, network in self.networks.items():
            for param in network.parameters():
                param.requires_grad = False

        with torch.set_grad_enabled(False):
            feature = self.networks['feature'](self.tensors['dataX'])

        return feature.cpu