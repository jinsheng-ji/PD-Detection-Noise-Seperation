# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.utils import seed_everything
from utils.loss import CenterLoss,batch_all_triplet_loss
from network.network import resnet50
    
def main():
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """
    Construct your Pytorch dataset for training here
    PRPD patterns are formated as images in our paper for PD type recognition
    """
    trainset = PD(img_size=input_size, is_train=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    # Define the CNN model
    model = resnet50(pretrained=True, num_classes=num_classes)
    model = torch.nn.DataParallel(model)
    model = model.to(device)
    model.train()

    # Define the loss and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    criterion_cent = CenterLoss(num_classes=num_classes, feat_dim=2, use_gpu=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=0.9, weight_decay=True)
    optimizer_centloss = torch.optim.SGD(criterion_cent.parameters(), lr=init_lr_cent)
    scheduler = CosineAnnealingLR(optimizer, epoch_num, eta_min=init_lr*0.01, last_epoch=-1)
    scheduler_ce = CosineAnnealingLR(optimizer_centloss, epoch_num, eta_min=init_lr_cent*0.1, last_epoch=-1)
    
    # Train
    for _ in range(1, epoch_num):
        for _, data in enumerate(trainloader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs,features,features_ce = model(images)
            loss_cl = criterion(outputs, labels)
            loss_ce = criterion_cent(features_ce, labels)
            loss_triplet = batch_all_triplet_loss(labels, F.normalize(features), margin=0.5)
            total_loss = loss_cl + 0.1*loss_ce + 0.3*loss_triplet   

            optimizer.zero_grad()
            optimizer_centloss.zero_grad()
            total_loss.backward()
            optimizer.step()
            optimizer_centloss.step()

        scheduler.step()
        scheduler_ce.step()
    
if __name__ == '__main__':
    seed = 1
    init_lr = 0.05
    init_lr_cent = 0.01
    num_classes = 4
    input_size, batch_size, epoch_num = 64, 64, 100
    main()
