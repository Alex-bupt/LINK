import torch
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from Params import args
from Model import Model, GaussianDiffusion, Denoise
from DataHandler import DataHandler
import numpy as np
from Utils.Utils import *
import os
import scipy.sparse as sp
import random
from scipy.sparse import coo_matrix


class Coach:
    def __init__(self, handler):
        self.handler = handler

        print('USER', args.user, 'ITEM', args.item)
        print('NUM OF INTERACTIONS', self.handler.trnLoader.dataset.__len__())
        self.metrics = dict()
        mets = ['Loss', 'Recall', 'NDCG']
        for met in mets:
            self.metrics['Train' + met] = list()
            self.metrics['Test' + met] = list()

    def prepareModel(self):
        entity_dim = args.latdim
        num_users = args.user
        num_items = args.item
        self.model = Model(entity_dim, num_users, num_items).cuda()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)

    def trainEpoch(self):
        trnLoader = self.handler.trnLoader
        epLoss, epRecLoss = 0, 0
        steps = trnLoader.dataset.__len__() // args.batch

        for i, (ancs, poss, negs) in enumerate(trnLoader):
            ancs = ancs.long().cuda()
            poss = poss.long().cuda()
            negs = negs.long().cuda()

            self.opt.zero_grad()
            usrEmbeds, itmEmbeds = self.model.forward(self.handler.torchBiAdj)
            ancEmbeds = usrEmbeds[ancs]
            posEmbeds = itmEmbeds[poss]
            negEmbeds = itmEmbeds[negs]

            scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
            bprLoss = -scoreDiff.sigmoid().log().sum() / args.batch
            regLoss = self.model.reg_loss() * args.reg
            loss = bprLoss + regLoss

            epRecLoss += bprLoss.item()
            epLoss += loss.item()

            loss.backward()
            self.opt.step()

            log('Step %d/%d: bpr : %.3f ; reg : %.3f' % (i, steps, bprLoss.item(), regLoss.item()), save=False, oneline=True)

        ret = dict()
        ret['Loss'] = epLoss / steps
        ret['BPR Loss'] = epRecLoss / steps
        return ret

    def testEpoch(self):
        tstLoader = self.handler.tstLoader
        epRecall, epNdcg = 0, 0
        num = tstLoader.dataset.__len__()
        steps = num // args.tstBat

        usrEmbeds, itmEmbeds = self.model.forward(self.handler.torchBiAdj)

        for usr, trnMask in tstLoader:
            usr = usr.long().cuda()
            trnMask = trnMask.cuda()
            allPreds = torch.mm(usrEmbeds[usr], torch.transpose(itmEmbeds, 1, 0)) * (1 - trnMask) - trnMask * 1e8
            _, topLocs = torch.topk(allPreds, args.topk)
            recall, ndcg = self.calcRes(topLocs.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr)
            epRecall += recall
            epNdcg += ndcg
            log('Steps %d/%d: recall = %.2f, ndcg = %.2f' % (usr, steps, recall, ndcg), save=False, oneline=True)

        ret = dict()
        ret['Recall'] = epRecall / num
        ret['NDCG'] = epNdcg / num
        return ret
