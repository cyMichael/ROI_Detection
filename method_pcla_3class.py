import os
import time
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.patch_classifier import instantiate_model



parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--model_load', type=str, default=None,
                        help='path to the wsi_classifier to load')
parser.add_argument('--data_folder', type=str,
                        help='path to the dataset')
parser.add_argument('--exp_name', type=str, default='pcla_3class',
                        help='name of this experiment')
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--validate', action='store_false', default=True, help='whether do validation')
parser.add_argument('--testing', action='store_false', default=True, help='whether do testing')
parser.add_argument('--n_epochs', type=int, default=20)
parser.add_argument('--lr_cla', type=float, default=5e-4, metavar='LR', help='learning rate')
parser.add_argument('--weight_decay_cla', type=float, default=1e-4, metavar='R', help='weight decay')
parser.add_argument('--patience', type=int, default=20, metavar='N_EPOCHS',
                        help='number of epochs (patience) for early stopping callback')
parser.add_argument('--save_model', action='store_false', default=True, help='toggle model saving')
parser.add_argument('--models_save_folder', type=str, default='./saved_models/',
                        help='folder in which to save model')
parser.add_argument('--num_class', type=int, default=3)

parser.add_argument(
    '-l', '--list',  # either of this switches
    nargs='+',       # one or more parameters to this switch
    type=float,        # /parameters/ are ints
    default=5e-4,
    dest='lst',      # store in 'lst'.
    default=[],      # since we're not specifying required.
)


args = parser.parse_args()
print(args)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def early_stopping(val_losses, patience):
    """ Return (True, min achieved val loss) if no val losses is under the minimal achieved val loss for patience
        epochs, otherwise (False, None) """
    # Do not stop until enough epochs have been made
    if len(val_losses) < patience:
        return False, None
    best_val_loss = np.min(val_losses)
    if not np.any(val_losses[-patience:] <= best_val_loss):
        return True, best_val_loss
    return False, None


def train_epoch(args, epoch, model, cla_optimizer, models_save_folder, data_loader, training=True):
       
    if training:
        model = model.train().to(device)
    else:
        model = model.eval().to(device)     
        
    cla_loss_fn = nn.CrossEntropyLoss()
        
    time_start = time.time()
    loss = []
    correct = 0
    total = 0
    for i, (image, label) in tqdm(enumerate(data_loader)):
        
        if training:
            cla_optimizer.zero_grad()   
        
        image, label = image.to(device), label.to(device)
        # Calculate classification loss
        logits = model(image)
        cla_loss = cla_loss_fn(logits,label)
        loss.append(cla_loss.item())
        
        if training:
            cla_loss.backward()
            cla_optimizer.step()
        
        _, Y_hat = torch.max(logits.data, 1)
        correct += sum(Y_hat==label)
        total += image.shape[0]
                        
    time_elapsed = time.time() - time_start
    
    mean_epoch_loss = np.mean(loss)
    std_batch_loss = np.std(loss)
    
    if args.save_model and training:
        save_path_cla = os.path.join(models_save_folder,
                             '{}_epoch{}_loss{:.3f}_acc{:.2f}.pt'.format(args.exp_name, epoch, mean_epoch_loss, correct.item()/total))
        torch.save(model.state_dict(), save_path_cla)

    if training:
        print('\ndoing training for {} took {} s, epoch={:.4f}+/-{:.4f}, acc={:.3f}'.format(epoch, time_elapsed, 
                            mean_epoch_loss, std_batch_loss, correct.item()/total))
    else:
        print('\ndoing testing for {} took {} s, epoch={:.4f}+/-{:.4f}, acc={:.3f}'.format(epoch, time_elapsed, 
                            mean_epoch_loss, std_batch_loss, correct.item()/total))
    return mean_epoch_loss, model


if __name__ == '__main__':    
    print('loading data')
    trans_train = transforms.Compose([
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.6632, 0.4123, 0.5529), (0.1618, 0.1749, 0.1478)),
        ])
    trans_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.6632, 0.4123, 0.5529), (0.1618, 0.1749, 0.1478)),
        ])
    trainset = datasets.ImageFolder(root = os.path.join(args.data_folder, 'train'), transform=trans_train)
    valset = datasets.ImageFolder(root = os.path.join(args.data_folder, 'val'), transform=trans_test)
    testset = datasets.ImageFolder(root = os.path.join(args.data_folder, 'test'), transform=trans_test)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last = True)
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    
    print('The class to idx map is:')
    print(trainset.class_to_idx)
    
    print('loading model checkpoint')
    # Load Models
    model, input_width = instantiate_model('vgg16', True, args.num_class)
    
    models_save_folder = args.models_save_folder
    
    os.makedirs(models_save_folder, exist_ok=True)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    if args.model_load:
        print('loading pre-trained models')
        model.load_state_dict(torch.load(os.path.join(args.models_save_folder, args.model_load)))
        
    
    cla_optimizer = optim.Adam(model.parameters(), lr=args.lr_cla,
                           weight_decay=args.weight_decay_cla)
    
    val_losses = []
    start_training_time = time.time()
    for epoch in range(args.start_epoch, args.n_epochs):
        # Train
        train_loss, model = train_epoch(args, epoch, model, cla_optimizer, 
                models_save_folder, train_loader)
    
        # Validate
        if args.validate:
            with torch.no_grad():
                val_loss, _ = train_epoch(args, epoch, model, cla_optimizer,  
                models_save_folder, 
                val_loader,
                training=False)
    
            # Early stopping
            val_losses.append(val_loss)
            do_stop, best_value = early_stopping(val_losses, patience=args.patience)
            if do_stop:
                print('Early stopping triggered: stopping training after no improvement on val set for '
                               '%d epochs with value %.3f' % (args.patience, best_value))
                break
            
        if args.testing:
            print('Starting testing...')
            with torch.no_grad():
                _, _ = train_epoch(args, epoch, model, cla_optimizer,  models_save_folder, test_loader, training=False)       
    print('Total training time %s' % (time.time() - start_training_time))