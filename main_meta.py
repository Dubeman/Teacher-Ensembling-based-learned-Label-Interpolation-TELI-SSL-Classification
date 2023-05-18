import argparse
import logging
import math
import os
import seaborn as sns
import matplotlib.patheffects as pe
import random
from datetime import datetime
import time
import cv2
import math
from mpl_toolkits.axes_grid1 import ImageGrid
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import wget
from google.colab.patches import cv2_imshow
from zipfile import ZipFile
import numpy as np
import torch
from torch import topk
from torchviz import make_dot
from torch.cuda import amp
from torch import nn
from torch.nn import functional as F
import shutil
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, ConcatDataset
from torch.utils.data import random_split
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from torch.autograd import Variable
import torchvision
from torchvision import models ,transforms
from manifold_models import resnet34, resnet101, resnet18, resnet50
from Grad_cam import GradCamModel
#import wandb
from tqdm import tqdm

from utils import train, weight_histograms
from data import DATASET_GETTERS, restructure_val_TinyImagenet
from models2 import VGG,ResNet9, MobileNet,ResNet101,ResNet34, WideResNet,build_wideresnet,NeuralNetwork, ResNet18,Cifar10CnnModel_2
from utils import (accuracy, evaluate, fit, plot_features,get_features,class_distribution_plotter)
from MAE_model import *
from MAE_utils import (setup_seed,test_evaluation)

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
#parser.add_argument('--name', type=str, required=True, help='experiment name')
parser.add_argument('--data-path', default='./data', type=str, help='data path')
parser.add_argument('--extract_tsne', default= False, type=str, help='whether or not TSNE plots are needed')
parser.add_argument('--load_ckp', default=  False, type=str, help='whether or not to resume training')
parser.add_argument('--teacher_1', default= 'MobileNet' , type=str, choices=['Res-18', 'MobileNet','MAE','VGG-11','VGG-13'], help='Pre-trained teacher 1 (on relevant dataset)')
parser.add_argument('--teacher_2', default=  'Res-18', type=str,choices=['Res-18', 'MobileNet','MAE','VGG-11','VGG-13'], help='Pre-trained teacher 2 (on relevant dataset)')
parser.add_argument('--dataset', default = 'FashionMNIST', type=str,
                    choices=['cifar10', 'cifar100','SVHN','tinyimagenet','MNIST','FashionMNIST'], help='dataset name')
parser.add_argument('--num-labeled', type=int, default=4000, help='number of labeled data')
parser.add_argument("--expand-labels", action="store_true", help="expand labels to fit eval steps")
parser.add_argument('--total-steps', default=300000, type=int, help='number of total steps to run')
parser.add_argument('--eval-step', default=1000, type=int, help='number of eval steps to run')
parser.add_argument('--start-step', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--workers', default=4, type=int, help='number of workers')
parser.add_argument('--num-classes', default=10, type=int, help='number of classes')
parser.add_argument('--resize', default=8, type=int, help='resize image')
parser.add_argument('--batch-size', default=1000, type=int, help='train batch size')
parser.add_argument('--teacher-dropout', default=0, type=float, help='dropout on last dense layer')
parser.add_argument('--student-dropout', default=0, type=float, help='dropout on last dense layer')
parser.add_argument('--teacher_lr', default=0.01, type=float, help='train learning late')
parser.add_argument('--student_lr', default=0.01, type=float, help='train learning late')
parser.add_argument('--momentum', default=0.9, type=float, help='SGD Momentum')
parser.add_argument('--nesterov', action='store_true', help='use nesterov')
parser.add_argument('--weight-decay', default=0, type=float, help='train weight decay')
parser.add_argument('--alpha', type=int, default=0.1, help='the alpha for the beta distribution mixup')
parser.add_argument('--ema', default=0, type=float, help='EMA decay rate')
parser.add_argument('--warmup-steps', default=0, type=int, help='warmup steps')
parser.add_argument('--student-wait-steps', default=0, type=int, help='warmup steps')
parser.add_argument('--grad-clip', default=1e9, type=float, help='gradient norm clipping')
parser.add_argument('--resume', default='', type=str, help='path to checkpoint')
parser.add_argument('--evaluate', action='store_true', help='only evaluate model on validation set')
parser.add_argument('--finetune', action='store_true',
                    help='only finetune model on labeled dataset')
parser.add_argument('--finetune-epochs', default=625, type=int, help='finetune epochs')
parser.add_argument('--finetune-batch-size', default=32, type=int, help='finetune batch size')
parser.add_argument('--finetune-lr', default=3e-5, type=float, help='finetune learning late')
parser.add_argument('--finetune-weight-decay', default=0, type=float, help='finetune weight decay')
parser.add_argument('--finetune-momentum', default=0.9, type=float, help='finetune SGD Momentum')
parser.add_argument('--seed', default=42, type=int, help='seed for initializing training')
parser.add_argument('--label-smoothing', default=0, type=float, help='label smoothing alpha')
parser.add_argument('--mu', default=1, type=int, help='coefficient of unlabeled batch size')
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

def save_ckp(state, save_path):
  torch.save(state, save_path)
    

def load_ckp(checkpoint_fpath, model):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
  #  optimizer.load_state_dict(checkpoint['optimizer'])
    return model, checkpoint['epoch']



def cam_extractor(student_model,dataset):
  # get the softmax weight
        params = list(student_model.parameters())
        weight_softmax = np.squeeze(params[-2].cpu().data.numpy())
        count = 0 
        for image,_ in dataset:
          plt.imshow(image.permute(1,2,0))
          plt.savefig('/content/Original/{}.jpg'.format(count))
          image_tensor = image.unsqueeze(0)
    # forward pass through model
          outputs = student_model(image_tensor.to(device))
    # get the softmax probabilities
          probs = F.softmax(outputs).data.squeeze()
    # get the class indices of top k probabilities
          class_idx = topk(probs, 1)[1].int()
    
    # generate class activation mapping for the top1 prediction
          CAMs = returnCAM(features_blobs[0], weight_softmax, class_idx)
    # file name to save the resulting CAM image with
          save_name = "{}".format(count)
    # show and save the results
          show_cam(CAMs, 64, 64, image.numpy(), class_idx, save_name)
          count+= 1





def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

def show_cam(CAMs, width, height, orig_image, class_idx, save_name):
    for i, cam in enumerate(CAMs):
        heatmap = cv2.applyColorMap(cv2.resize(cam,(width, height)), cv2.COLORMAP_JET)
        
        result = heatmap * 0.5  #+ orig_image * 0.5
        # put class label text on the result
        cv2.putText(result, str(int(class_idx[i])), (5, 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        plt.imshow(result/255.)
        plt.savefig(f"outputs/CAM_{save_name}.jpg")
        cv2.waitKey(0)
       # cv2.imwrite(f"outputs/CAM_{save_name}.jpg", result)

def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

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

def evaluate(model, val_loader,teacher_inputs,device):
    model.eval()
    outputs = []
    with torch.no_grad():
      for images,labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs.append(model.validation_step(images,labels,teacher_inputs,device)) 
    return model.validation_epoch_end(outputs)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_cycle(device,unsupervised_epochs,supervised_epochs, max_lr, model,model_1,model_2, labeled_train_loader,unlabeled_loader, val_loader,save_path, 
                  weight_decay=0, grad_clip=None, opt_func = torch.optim.SGD):

    np.seterr(invalid='ignore')
    set_seed(args)
    torch.cuda.empty_cache()
    
    history = []
    train_losses = []
    lrs = []
    alpha = []
    no_decay = ['bn']
    student_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
      ]
    optimizer_normal = opt_func(student_parameters,
                            lr=args.student_lr,
                            momentum=args.momentum,
                            nesterov=args.nesterov)

    optimizer_meta = opt_func(student_parameters,
                            lr=args.student_lr,
                            momentum=args.momentum,
                            nesterov=args.nesterov)                        


    model_1.eval()
    model_2.eval()
    if args.load_ckp:
      model, optimizer, start_epoch = load_ckp(save_path, model, optimizer)
      unsupervised_epochs = unsupervised_epochs - start_epoch
      supervised_epochs = 0
      
      for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
                    
    # Set up cutom optimizer with weight decay
   # optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # Set up one-cycle learning rate scheduler
 #   sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, 
 #                                               steps_per_epoch=len(train_loader))
    
    s_scheduler = get_cosine_schedule_with_warmup(optimizer_normal,
                                                  args.warmup_steps,
                                                  args.total_steps,
                                                  args.student_wait_steps)

    scaler_normal = torch.cuda.amp.GradScaler()
    scaler_lambda = torch.cuda.amp.GradScaler()

    supervised_phase(supervised_epochs,model,labeled_train_loader,val_loader,optimizer_normal,s_scheduler,scaler_normal,history,lrs,train_losses)
    
    print("----currently training on pseudo-labelled data----")
 
    
    for epoch in range(unsupervised_epochs):
        # Training Phase 
        model.train()
        for images_us,_ in unlabeled_loader:  # Total no of batches per epoch
          
          with torch.no_grad():
            images_us = images_us.to(device)
            t1_logits = model_1(images_us)
            t2_logits = model_2(images_us)
            soft_pseudo_label_1 = torch.softmax(t1_logits.detach() / args.temperature, dim=-1)
            soft_pseudo_label_2 = torch.softmax(t2_logits.detach() / args.temperature, dim=-1) 

            max_probs_1, hard_pseudo_label_1 = torch.max(soft_pseudo_label_1, dim=-1)
            max_probs_2, hard_pseudo_label_2 = torch.max(soft_pseudo_label_2, dim=-1)
            mask_1 = max_probs_1.ge(args.threshold).float()
            mask_2 = max_probs_2.ge(args.threshold).float()
          
      #   student_logits_us = model(images_us)
       #   student_logits_uw = model(images_uw)
        #  soft_pseudo_label = torch.softmax(student_logits_uw.detach() / args.temperature, dim=-1) # Slightly augmented version of the same image to take the unsupervised loss
        #  max_probs, hard_pseudo_label = torch.max(soft_pseudo_label, dim=-1) # hard_pseudo_label To be used later by the student
        #  mask = max_probs.ge(args.threshold).float()
       #   t_loss_u = torch.mean(-(soft_pseudo_label * torch.log_softmax(student_logits_us, dim=-1)).sum(dim=-1) * mask)
      #    weight_u = args.lambda_u * min(1., (i + 1) / args.uda_steps) 
         # loss_uda = weight_u * t_loss_u #UDA Loss

          
            mismatched_indices = torch.where(torch.logical_and(hard_pseudo_label_1 != hard_pseudo_label_2, torch.logical_and(mask_1 == 1, mask_2 == 1)))
            matched_indices = torch.where(torch.logical_and(hard_pseudo_label_1 == hard_pseudo_label_2, torch.logical_and(mask_1 == 1, mask_2 == 1)))
            matched_images = images_us[matched_indices[0]].to(device)
            mismatched_images = images_us[mismatched_indices[0]].to(device)
            matched_labels = torch.take(hard_pseudo_label_1,matched_indices[0]).to(device)
            mixup_labels_1 = torch.take(hard_pseudo_label_1,mismatched_indices[0]).to(device)
            mixup_labels_2 = torch.take(hard_pseudo_label_2,mismatched_indices[0]).to(device)
          
          (teacher_inputs,_) = teacher_input_concatenator(model_1,model_2,labeled_loader,device)
          teacher_inputs += torch.randn(teacher_inputs.shape).to(device)
        #  teacher_inputs.to(device)
          with torch.cuda.amp.autocast():
            if matched_images.shape[0] == 0:
             (loss_mixup,lam) = model.training_step_label_mixup(mismatched_images, mixup_labels_1,mixup_labels_2,teacher_inputs,device)
            
             loss_normal = torch.zeros(1,1).to(device)
             

            elif mismatched_images.shape[0] == 0:
             
             loss_normal = model.training_step_normal(matched_images, matched_labels,teacher_inputs) 
             loss_mixup = torch.zeros(1,1).to(device)
             
            else:
              (loss_mixup,lam) = model.training_step_label_mixup(mismatched_images, mixup_labels_1,mixup_labels_2, teacher_inputs,device)
            # loss_mixup = model.training_step_label_mixup(mismatched_inputs, mixup_labels_1,mixup_labels_2,alpha)

              loss_normal = model.training_step_normal(matched_images, matched_labels,teacher_inputs,device)
             
              # Normal Pseudo-mixup loss 
            loss_total =loss_normal + 0.1*loss_mixup
            scaler_normal.scale(loss_total).backward()
            scaler_normal.step(optimizer_normal)
            scaler_normal.update()

               # Meta lambda loss on
            
            loss_meta = model.training_step_NeuralNetwork(model,val_loader,teacher_inputs, device)
            scaler_lambda.scale(loss_meta).backward()
            scaler_lambda.step(optimizer_meta)
            scaler_lambda.update()

            optimizer_normal.zero_grad()
            optimizer_meta.zero_grad()  
          
            train_losses.append(loss_total.reshape([]))
          
            
            # Gradient clipping
          if grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
            # Record & update learning rate
          lrs.append(get_lr(optimizer_normal))
          s_scheduler.step()
          
        # Validation phase
        
        print("normal loss after {} epochs : {}".format(epoch,loss_normal.item()))

        print("mixup loss after {} epochs: {}".format(epoch,loss_mixup.item()))
        
        print("last updated lambda= {}".format(lam))
        
        result = evaluate(model, val_loader,teacher_inputs,device)
         

        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        print("---------")
        history.append(result)

     
    return history

def supervised_phase(supervised_epochs,model,labeled_train_loader,val_loader,optimizer_normal,s_scheduler,scaler_normal,history,lrs,train_losses):
    print("---Currently training over labelled data---")
    for epoch in range(supervised_epochs):
                # Training Phase 
        model.train()

        for i,(images,labels) in enumerate(labeled_train_loader):
            
            images = images.to(device)
            labels = labels.to(device)


            loss = model.training_step_normal(images,labels)
            train_losses.append(loss.reshape([]))
            scaler_normal.scale(loss).backward()
            scaler_normal.step(optimizer_normal)
            scaler_normal.update()
            # Gradient clipping
            if grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
            
            optimizer_normal.zero_grad()
            
            
        # Validation phase
       

        
        # Record & update learning rate
        lrs.append(get_lr(optimizer_normal))
        s_scheduler.step()
        
          
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
      
def teacher_input_concatenator(model_1,model_2,labeled_loader,device):
  for images,_ in labeled_loader:
    t1_logits = model_1(images.to(device))
    t2_logits = model_2(images.to(device))
    tens = torch.cat((t1_logits, t2_logits), 0)
    meta_tensor = torch.flatten(tens)
    break
  
  return (meta_tensor,meta_tensor.shape[0])    
        
              
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
        

        
       # model_1 = torch.load('/content/gdrive/MyDrive/Psuedo-Mixup/teacher models/MAE_cifar_10_31epochs.pt',map_location=torch.device('cuda'))
        #mode =  resnet101(num_classes=10, dropout = False)
       # model_2 = VGG('VGG13',10) 
        
       # model_2.load_state_dict(torch.load('/content/gdrive/MyDrive/Psuedo-Mixup/teacher models/model_1_cifar10',map_location=torch.device('cuda')))
        
        if torch.cuda.is_available():
          device = torch.device("cuda:0")
        else:
          device = torch.device("cpu")

        print("using device : {}".format(device)) 
        
        
      #  model_1 =  resnet101(num_classes=10, dropout = False).to(device)
     #   model_2 =  resnet50(num_classes=10, dropout = False).to(device)
        
        
        
        
        train_sampler = RandomSampler #if args.local_rank == -1 else DistributedSampler
        labeled_loader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=2,
        drop_last=True)

        unlabeled_loader = DataLoader(
        unlabeled_dataset,
        sampler=train_sampler(unlabeled_dataset),
        batch_size=args.batch_size * args.mu,
        num_workers=2,
        drop_last=True)

        finetune_loader = DataLoader(
        finetune_dataset,
        sampler=train_sampler(finetune_dataset),
        batch_size=args.batch_size,
        num_workers=2,
        drop_last=True)
        
        test_loader = DataLoader(test_dataset,
                             sampler=RandomSampler(test_dataset),
                             batch_size=args.batch_size,
                             num_workers=2)
        
        
        
        
        if args.teacher_1 == "MobileNet" and args.teacher_2 == "Res-18" and args.dataset == 'FashionMNIST':
          model_1 = MobileNet(1,10).to(device)
          model_1.load_state_dict(torch.load('/content/gdrive/MyDrive/Psuedo-Mixup/teacher models/100_MNIST_Mobilenet.pth',map_location=torch.device('cuda')))

          model_2 = resnet18(num_classes = 10,in_channels = 1).to(device)
          model_2.load_state_dict(torch.load('/content/gdrive/MyDrive/Psuedo-Mixup/teacher models/400_mnist_res18.pth',map_location=torch.device('cuda')))
          
          (meta_tensor,meta_input_shape) = teacher_input_concatenator(model_1,model_2,labeled_loader,device)
          
          student_model = ResNet9(1,10,meta_input_shape,meta_tensor).to(device)
        
        
        if args.teacher_1 == 'VGG-13' and args.teacher_2 == 'MAE' and args.dataset == 'cifar10':
          model_1 = VGG('VGG13',3,10).to(device)
          model_1.load_state_dict(torch.load('/content/gdrive/MyDrive/Psuedo-Mixup/teacher models/model_1_cifar10',map_location=torch.device('cuda')))
       
          model_2 = torch.load('/content/gdrive/MyDrive/Psuedo-Mixup/teacher models/MAE_cifar_10_31epochs.pt',map_location=torch.device('cuda')).to(device) 
          
          (meta_tensor,meta_input_shape) = teacher_input_concatenator(model_1,model_2,labeled_loader,device)

          student_model = Cifar10CnnModel_2(3,10,meta_input_shape,meta_tensor).to(device)

        
        if args.teacher_1 == 'VGG-11' and args.teacher_2 == 'VGG-13' and args.dataset == 'cifar10':
          model_1 = VGG('VGG11',3,10).to(device)
          model_1.load_state_dict(torch.load('/content/gdrive/MyDrive/Psuedo-Mixup/teacher models/model_2_cifar10',map_location=torch.device('cuda')))
       
          model_2 = VGG('VGG13',3,10).to(device) 
          model_2.load_state_dict(torch.load('/content/gdrive/MyDrive/Psuedo-Mixup/teacher models/model_1_cifar10',map_location=torch.device('cuda')))
          
          (meta_tensor,meta_input_shape) = teacher_input_concatenator(model_1,model_2,labeled_loader,device)
          student_model = ResNet9(3,10,meta_input_shape,meta_tensor).to(device)

        
        

      
       


        max_lr = 0.01
        grad_clip = 0.1
        weight_decay = 1e-4
        opt_func = optim.SGD
        
        


        
        
      #  student_model = MobileNet(10).to(device)
      #  student_model,epochs = load_ckp('/content/gdrive/MyDrive/Psuedo-Mixup/Student models/MNIST_res18_mobilenet_mobilenet_learned_alpha.pth' , student_model)
      #  student_model.load_state_dict(torch.load('/content/gdrive/MyDrive/Psuedo-Mixup/Student models/MNIST_res18_mobilenet_mobilenet_learned_alpha.pth',map_location=torch.device('cpu')))
      #  print(student_model)
  
        
        
         
      #  print(student_model)
      
        save_path = '/content/gdrive/MyDrive/Psuedo-Mixup/Student models/something.pth'
      #  if os.path.exists(save_path):
      #    args.load_ckp =  True

        history = [evaluate(student_model, finetune_loader,meta_tensor,device)]
        history += fit_one_cycle(device = device,unsupervised_epochs =20,supervised_epochs = 0, max_lr = max_lr,model = student_model,model_1 = model_1,model_2 = model_2, labeled_train_loader = labeled_loader,unlabeled_loader = unlabeled_loader, val_loader = finetune_loader, 
                weight_decay=0, grad_clip=None, opt_func=opt_func,save_path = save_path)
        
       
        plot_losses(history)
        plot_accuracies(history)
        test = evaluate(student_model,test_loader)
        print(test)
        
