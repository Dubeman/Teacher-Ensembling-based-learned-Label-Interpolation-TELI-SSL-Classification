import argparse
import logging
import math
import os
import random
import time
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import wget
from zipfile import ZipFile
import numpy as np
import torch
from torch.cuda import amp
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, ConcatDataset
from torch.utils.data import random_split
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
#import wandb
from tqdm import tqdm


from data import DATASET_GETTERS, restructure_val_TinyImagenet
from models import WideResNet, ModelEMA,ImageClassificationBaseMixup,VGG,ResNet9, MobileNet,ResNet101,ResNet34
from utils import (accuracy, evaluate, fit)


logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
#parser.add_argument('--name', type=str, required=True, help='experiment name')
parser.add_argument('--data-path', default='./data', type=str, help='data path')
parser.add_argument('--save-path', default='./checkpoint', type=str, help='save path')
parser.add_argument('--dataset', default = 'cifar10', type=str,
                    choices=['cifar10', 'cifar100','SVHN','tinyimagenet'], help='dataset name')
parser.add_argument('--num-labeled', type=int, default=4000, help='number of labeled data')
parser.add_argument("--expand-labels", action="store_true", help="expand labels to fit eval steps")
parser.add_argument('--total-steps', default=300000, type=int, help='number of total steps to run')
parser.add_argument('--eval-step', default=1000, type=int, help='number of eval steps to run')
parser.add_argument('--start-step', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--workers', default=4, type=int, help='number of workers')
parser.add_argument('--num-classes', default=10, type=int, help='number of classes')
parser.add_argument('--resize', default=32, type=int, help='resize image')
parser.add_argument('--batch-size', default=64, type=int, help='train batch size')
parser.add_argument('--teacher-dropout', default=0, type=float, help='dropout on last dense layer')
parser.add_argument('--student-dropout', default=0, type=float, help='dropout on last dense layer')
parser.add_argument('--teacher_lr', default=0.01, type=float, help='train learning late')
parser.add_argument('--student_lr', default=0.1, type=float, help='train learning late')
parser.add_argument('--momentum', default=0.9, type=float, help='SGD Momentum')
parser.add_argument('--nesterov', action='store_true', help='use nesterov')
parser.add_argument('--weight-decay', default=0, type=float, help='train weight decay')
parser.add_argument('--alpha', type=int, default=1, help='the alpha for the beta distribution mixup')
parser.add_argument('--ema', default=0, type=float, help='EMA decay rate')
parser.add_argument('--warmup-steps', default=0, type=int, help='warmup steps')
parser.add_argument('--student-wait-steps', default=0, type=int, help='warmup steps')
parser.add_argument('--grad-clip', default=1e9, type=float, help='gradient norm clipping')
parser.add_argument('--resume', default='', type=str, help='path to checkpoint')
parser.add_argument('--evaluate', action='store_true', help='only evaluate model on validation set')
parser.add_argument('--finetune', action='store_true',
                    help='only finetune model on labeled dataset')
parser.add_argument('--finetune-epochs', default=625, type=int, help='finetune epochs')
parser.add_argument('--finetune-batch-size', default=512, type=int, help='finetune batch size')
parser.add_argument('--finetune-lr', default=3e-5, type=float, help='finetune learning late')
parser.add_argument('--finetune-weight-decay', default=0, type=float, help='finetune weight decay')
parser.add_argument('--finetune-momentum', default=0.9, type=float, help='finetune SGD Momentum')
parser.add_argument('--seed', default=42, type=int, help='seed for initializing training')
parser.add_argument('--label-smoothing', default=0, type=float, help='label smoothing alpha')
parser.add_argument('--mu', default=2, type=int, help='coefficient of unlabeled batch size')
parser.add_argument('--threshold', default=0.95, type=float, help='pseudo label threshold')
parser.add_argument('--temperature', default=1, type=float, help='pseudo label temperature')
parser.add_argument('--lambda-u', default=1, type=float, help='coefficient of unlabeled loss')
parser.add_argument('--uda-steps', default=1, type=float, help='warmup steps of lambda-u')
parser.add_argument("--randaug", nargs="+", type=int, help="use it like this. --randaug 2 10")
parser.add_argument("--amp", action="store_true", help="use 16-bit (mixed) precision")
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument("--local_rank", type=int, default=-1,
                    help="For distributed training: local_rank")
args = parser.parse_args(args=[])

def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

@torch.no_grad()
def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_wait_steps=0,
                                    num_cycles=0.5,
                                    last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_wait_steps:
            return 0.0

        if current_step < num_warmup_steps + num_wait_steps:
            return float(current_step) / float(max(1, num_warmup_steps + num_wait_steps))

        progress = float(current_step - num_warmup_steps - num_wait_steps) / \
            float(max(1, num_training_steps - num_warmup_steps - num_wait_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def evaluate(model, val_loader):
    model.eval()
    outputs = []
    for images,labels in val_loader:
      #images = images.to(device)
      #labels = labels.to(device)
      outputs.append(model.validation_step(images,labels)) 
    return model.validation_epoch_end(outputs)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_cycle(unsupervised_epochs,supervised_epochs, max_lr, model,model_1,model_2, labeled_loader,unlabeled_loader, val_loader, 
                  weight_decay=0, grad_clip=None, opt_func = torch.optim.SGD):
    
    set_seed(args)
    torch.cuda.empty_cache()
    history = []
    train_losses = []
    lrs = []
    no_decay = ['bn']
    student_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
      ]
    optimizer = opt_func(student_parameters,
                            lr=args.student_lr,
                            momentum=args.momentum,
                            nesterov=args.nesterov)


    # Set up cutom optimizer with weight decay
   # optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # Set up one-cycle learning rate scheduler
 #   sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, 
 #                                               steps_per_epoch=len(train_loader))
    s_scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                  args.warmup_steps,
                                                  args.total_steps,
                                                  args.student_wait_steps)

    

    
    print("---Currently training over labelled data---")
    for epoch in range(supervised_epochs):
        # Training Phase 
        model.train()
        

        for images,labels in labeled_train_loader:
            images = images.to(device)
            labels = labels.to(device)
            loss = model.training_step_normal(images,labels)
            train_losses.append(loss)
            loss.backward()
            
            # Gradient clipping
            if grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
            optimizer.step()
            optimizer.zero_grad()
            
            # Record & update learning rate
            lrs.append(get_lr(optimizer))
            s_scheduler.step()
        
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
 #   return history


    
    print("----currently training on pseudo-labelled data----")
    
    
    for epoch in range(unsupervised_epochs):
        # Training Phase 
     #   model.train()
      #  train_losses = []
      #  lrs = []
        if (epoch+1)%10 == 0:
          for images,labels in labeled_train_loader:
            images =images.to(device)
            labels = labels.to(device)
            loss = model.training_step_normal(images,labels)
            train_losses.append(loss)
            loss.backward()
            
            # Gradient clipping
            if grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
            optimizer.step()
            optimizer.zero_grad()
            
            # Record & update learning rate
            lrs.append(get_lr(optimizer))
            s_scheduler.step()
            
        
        # Validation phase
          result = evaluate(model, val_loader)
          result['train_loss'] = torch.stack(train_losses).mean().item()
          result['lrs'] = lrs
          model.epoch_end(epoch, result)
          history.append(result) 

        unlabeled_iter = iter(unlabeled_loader)
        count_batches =0
        for (images,_),_ in unlabeled_loader:
          for j in images:
            count_batches +=1
        for i in range(count_batches):  # Total no of batches per epoch
          
          
          (images_us,images_uw), _ = next(unlabeled_iter)
         # images_us = images_us.to(device)
         # images_uw = images_uw.to(device)
          t1_logits = model_1(images_us)
          t2_logits = model_2(images_us)
          soft_pseudo_label_1 = torch.softmax(t1_logits.detach() / args.temperature, dim=-1)
          soft_pseudo_label_2 = torch.softmax(t2_logits.detach() / args.temperature, dim=-1) 

          max_probs_1, hard_pseudo_label_1 = torch.max(soft_pseudo_label_1, dim=-1)
          max_probs_2, hard_pseudo_label_2 = torch.max(soft_pseudo_label_2, dim=-1)
          
          
       #   student_logits_us = model(images_us)
       #   student_logits_uw = model(images_uw)
        #  soft_pseudo_label = torch.softmax(student_logits_uw.detach() / args.temperature, dim=-1) # Slightly augmented version of the same image to take the unsupervised loss
        #  max_probs, hard_pseudo_label = torch.max(soft_pseudo_label, dim=-1) # hard_pseudo_label To be used later by the student
        #  mask = max_probs.ge(args.threshold).float()
       #   t_loss_u = torch.mean(-(soft_pseudo_label * torch.log_softmax(student_logits_us, dim=-1)).sum(dim=-1) * mask)
      #    weight_u = args.lambda_u * min(1., (i + 1) / args.uda_steps) 
         # loss_uda = weight_u * t_loss_u #UDA Loss

          
          mismatched_indices = torch.where(hard_pseudo_label_1 != hard_pseudo_label_2)
          matched_indices = torch.where(hard_pseudo_label_1 == hard_pseudo_label_2)
          
          mismatched_list = [images_us[i] for i in mismatched_indices[0].numpy()]
          
          matched_list = [images_us[i] for i in matched_indices[0].numpy()]
          print("len(matched_list) :- {}".format(len(matched_list)))
          print("len(mismatched_list) :- {}".format(len(mismatched_list)))

          if len(matched_list) ==0 or len(mismatched_list) ==0:
            if len(matched_list) == 0:
              mismatched_inputs = torch.stack(mismatched_list)
              mismatched_inputs = mismatched_inputs.to(device)

            else:
              matched_inputs = torch.stack(matched_list)
              matched_inputs = matched_inputs.to(device)

          else :
            mismatched_inputs = torch.stack(mismatched_list)
            mismatched_inputs = mismatched_inputs.to(device)
            matched_inputs = torch.stack(matched_list)
            matched_inputs = matched_inputs.to(device)

          
          matched_labels = torch.take(hard_pseudo_label_1,matched_indices[0])
          matched_labels = matched_labels.to(device)

          
          

          mixup_labels_1 = torch.take(hard_pseudo_label_1,mismatched_indices[0])
          
          mixup_labels_2 = torch.take(hard_pseudo_label_2,mismatched_indices[0])
          print("len(mixup_labels_1: {}".format(len(mixup_labels_1)))
          print("len(mixup_labels_2: {}".format(len(mixup_labels_2)))

          mixup_labels_1 = mixup_labels_1.to(device)
          mixup_labels_2 = mixup_labels_2.to(device)
         
          if len(matched_indices)==0:
            loss_mixup = model.training_step_label_mixup(mismatched_inputs, mixup_labels_1,mixup_labels_2,alpha = 1)
            loss_normal = torch.zeros(2,3)

          elif len(mismatched_indices) == 0:
            loss_normal = model.training_step_normal(matched_inputs, matched_labels) 
            loss_mixup = torch.zeros(2,3)
           
          else:
            loss_mixup = model.training_step_label_mixup(mismatched_inputs, mixup_labels_1,mixup_labels_2,args.alpha)
            loss_normal = model.training_step_normal(matched_inputs, matched_labels)
          

          loss_total = loss_normal + 0.1*loss_mixup  # loss_uda
          
          train_losses.append(loss_total)
          loss_total.backward()
            
            # Gradient clipping
          if grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
          optimizer.step()
          optimizer.zero_grad()
            
          # Record & update learning rate
          lrs.append(get_lr(optimizer))
          s_scheduler.step()
        
        # Validation phase
        
        
        
        print("normal loss after {} epochs : {}".format(epoch,loss_normal.item()))

        print("mixup loss after {} epochs: {}".format(epoch,loss_mixup.item()))
        
        result = evaluate(model, val_loader)
        
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        print("---------")
        history.append(result)
    return history
def supervised_phase(labeled_batch,model,labeled_train_loader,optimizer,s_scheduler,history):
   print("---Currently training over labelled data---")
   images,labels = batch
   images = images.to(device)
   labels = labels.to(device)
   loss = model.training_step_normal(images,labels)
   train_losses.append(loss)
   loss.backward()
            
            # Gradient clipping
   if grad_clip: 
    nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
   optimizer.step()
   optimizer.zero_grad()
            
            # Record & update learning rate
   lrs.append(get_lr(optimizer))
   s_scheduler.step()
        
   return history
        
def mixed_fit(supervised_epochs,mixed_epochs, max_lr, model, labeled_train_loader,unlabeled_loader, val_loader, 
                  weight_decay=0, grad_clip=None, opt_func = torch.optim.SGD):
    set_seed(args)
    torch.cuda.empty_cache()
    history = []
    train_losses = []
    lrs = []
    no_decay = ['bn']
    student_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
      ]
    optimizer = opt_func(student_parameters,
                            lr=args.student_lr,
                            momentum=args.momentum,
                            nesterov=args.nesterov)


    # Set up cutom optimizer with weight decay
   # optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # Set up one-cycle learning rate scheduler
 #   sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, 
 #                                               steps_per_epoch=len(train_loader))
    s_scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                  args.warmup_steps,
                                                  args.total_steps,
                                                  args.student_wait_steps)
  
    unlabeled_iter = iter(unlabeled_loader)
    labeled_iter = iter(labeled_train_loader)
    for epochs in range(mixed_epochs):
      k = random.randint(0, 1)
      if k == 1:
        batch = batch_sampler(labeled_iter)
        if batch is None :
          
     
          
          
          model.train()
        

          
          images = images.to(device)
          labels = labels.to(device)
          loss = model.training_step_normal(images,labels)
          train_losses.append(loss)
          loss.backward()
            
            # Gradient clipping
          if grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
          optimizer.step()
          optimizer.zero_grad()
            
            # Record & update learning rate
          lrs.append(get_lr(optimizer))
          s_scheduler.step()
      
      
      
      
      
      
      
      
      
      else :
        
          (images_us,images_uw), _ = next(unlabeled_iter)
         # images_us = images_us.to(device)
         # images_uw = images_uw.to(device)
          t1_logits = model_1(images_us)
          t2_logits = model_2(images_us)
          soft_pseudo_label_1 = torch.softmax(t1_logits.detach() / args.temperature, dim=-1)
          soft_pseudo_label_2 = torch.softmax(t2_logits.detach() / args.temperature, dim=-1) 

          max_probs_1, hard_pseudo_label_1 = torch.max(soft_pseudo_label_1, dim=-1)
          max_probs_2, hard_pseudo_label_2 = torch.max(soft_pseudo_label_2, dim=-1)
          
          
       #   student_logits_us = model(images_us)
       #   student_logits_uw = model(images_uw)
        #  soft_pseudo_label = torch.softmax(student_logits_uw.detach() / args.temperature, dim=-1) # Slightly augmented version of the same image to take the unsupervised loss
        #  max_probs, hard_pseudo_label = torch.max(soft_pseudo_label, dim=-1) # hard_pseudo_label To be used later by the student
        #  mask = max_probs.ge(args.threshold).float()
       #   t_loss_u = torch.mean(-(soft_pseudo_label * torch.log_softmax(student_logits_us, dim=-1)).sum(dim=-1) * mask)
      #    weight_u = args.lambda_u * min(1., (i + 1) / args.uda_steps) 
         # loss_uda = weight_u * t_loss_u #UDA Loss

          
          mismatched_indices = torch.where(hard_pseudo_label_1 != hard_pseudo_label_2)
          matched_indices = torch.where(hard_pseudo_label_1 == hard_pseudo_label_2)
          
          mismatched_list = [images_us[i] for i in mismatched_indices[0].numpy()]
          
          matched_list = [images_us[i] for i in matched_indices[0].numpy()]
          print("len(matched_list) :- {}".format(len(matched_list)))
          print("len(mismatched_list) :- {}".format(len(mismatched_list)))

          if len(matched_list) ==0 or len(mismatched_list) ==0:
            if len(matched_list) == 0:
              mismatched_inputs = torch.stack(mismatched_list)
              mismatched_inputs = mismatched_inputs.to(device)

            else:
              matched_inputs = torch.stack(matched_list)
              matched_inputs = matched_inputs.to(device)

          else :
            mismatched_inputs = torch.stack(mismatched_list)
            mismatched_inputs = mismatched_inputs.to(device)
            matched_inputs = torch.stack(matched_list)
            matched_inputs = matched_inputs.to(device)

          
          matched_labels = torch.take(hard_pseudo_label_1,matched_indices[0])
          matched_labels = matched_labels.to(device)

          
          

          mixup_labels_1 = torch.take(hard_pseudo_label_1,mismatched_indices[0])
          
          mixup_labels_2 = torch.take(hard_pseudo_label_2,mismatched_indices[0])
          print("len(mixup_labels_1: {}".format(len(mixup_labels_1)))
          print("len(mixup_labels_2: {}".format(len(mixup_labels_2)))

          mixup_labels_1 = mixup_labels_1.to(device)
          mixup_labels_2 = mixup_labels_2.to(device)
         
          if len(matched_indices)==0:
            loss_mixup = model.training_step_label_mixup(mismatched_inputs, mixup_labels_1,mixup_labels_2,alpha = 1)
            loss_normal = torch.zeros(2,3)

          elif len(mismatched_indices) == 0:
            loss_normal = model.training_step_normal(matched_inputs, matched_labels) 
            loss_mixup = torch.zeros(2,3)
           
          else:
            loss_mixup = model.training_step_label_mixup(mismatched_inputs, mixup_labels_1,mixup_labels_2,args.alpha)
            loss_normal = model.training_step_normal(matched_inputs, matched_labels)
          

          loss_total = loss_normal + 0.1*loss_mixup  # loss_uda
          
          train_losses.append(loss_total)
          loss_total.backward()
            
            # Gradient clipping
          if grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
          optimizer.step()
          optimizer.zero_grad()
            
          # Record & update learning rate
          lrs.append(get_lr(optimizer))
          s_scheduler.step()
        
        # Validation phase
        
        
        
      print("normal loss after {} epochs : {}".format(epoch,loss_normal.item()))

      print("mixup loss after {} epochs: {}".format(epoch,loss_mixup.item()))
        
      result = evaluate(model, val_loader)
        
      result['train_loss'] = torch.stack(train_losses).mean().item()
      result['lrs'] = lrs
      model.epoch_end(epoch, result)
      print("---------")
      history.append(result)
    return history



def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')
    plt.savefig('SVHN Loss')


def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')
    plt.savefig('SVHN Accuracy')

if __name__ == '__main__':
          
    
    

        
        labeled_dataset, unlabeled_dataset, test_dataset, finetune_dataset = DATASET_GETTERS[args.dataset](args)
       
        
        model_1 = VGG('VGG13') 
        model_2 = VGG('VGG11')
        model_1.load_state_dict(torch.load('/content/drive/MyDrive/Psuedo-Mixup/model_1_cifar100',map_location=torch.device('cpu')))
        model_2.load_state_dict(torch.load('/content/drive/MyDrive/Psuedo-Mixup/model_2_cifar100',map_location=torch.device('cpu')))

        if torch.cuda.is_available():
          device = torch.device("cuda:0")
        else:
          device = torch.device("cpu")

        print("using device : {}".format(device)) 
        
        train_sampler = RandomSampler #if args.local_rank == -1 else DistributedSampler
        labeled_loader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.workers,
        drop_last=True)

        unlabeled_loader = DataLoader(
        unlabeled_dataset,
        sampler=train_sampler(unlabeled_dataset),
        batch_size=args.batch_size * args.mu,
        num_workers=args.workers,
        drop_last=True)

        test_loader = DataLoader(test_dataset,
                             sampler=RandomSampler(test_dataset),
                             batch_size=args.batch_size,
                             num_workers=args.workers)

        for epoch in range(2):
          print("--epoch {}--".format(epoch))
          unlabeled_iter = iter(unlabeled_loader)
          count = 0
          for (images_us,_),_ in next(unlabeled_iter):
            count +=1 
            print(count)

                
        max_lr = 0.01
        grad_clip = 0.1
        weight_decay = 1e-4

        val_size = 1000
        train_size = len(labeled_dataset) - val_size
        train_ds, val_ds = random_split(labeled_dataset, [train_size, val_size])

        labeled_train_loader = DataLoader(train_ds, batch_size= args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_dl = DataLoader(val_ds, args.batch_size*2, num_workers=2, pin_memory=True)
        
        
        opt_func = optim.SGD
        #model_1 = ResNet101().to(device)
        #model_2 = ResNet34().to(device)
        student_model = ResNet9(3,10).to(device)
        FEATS = []
        features = {}
        hook = student_model.classifier[1].register_forward_hook(get_features('feats'))                     