import os
import time
import torch
import h5py
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets.dataset_h5 import Whole_Slide_BagV2
from models.patch_classifier import instantiate_model



parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--exp_name', type=str, default='pcla_3class',help='experiment code for saving results')
parser.add_argument('--model_load', type=str,
                        help='path to the wsi_classifier to load')
parser.add_argument('--csv_path', type=str,
                        help='name of csv file that contains WSI slide ids')
parser.add_argument('--patch_path', type=str,
                        help='path to the patches')
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--results_dir', type=str,
                        help='folder in which to save model')
parser.add_argument('--classification_save_dir', type=str,
                        help='folder in which to save model')
parser.add_argument('--models_save_folder', type=str, default='./saved_models/',
                        help='folder in which to save model')
parser.add_argument('--num_class', type=int, default=3)
parser.add_argument('--auto_skip', default=False, action='store_true')

parser.add_argument('--mean1', type=float, default=5e-4)
parser.add_argument('--mean2', type=float, default=5e-4)
parser.add_argument('--mean3', type=float, default=5e-4)

parser.add_argument('--std1', type=float, default=5e-4)
parser.add_argument('--std2', type=float, default=5e-4)
parser.add_argument('--std3', type=float, default=5e-4)

args = parser.parse_args()
args.results_dir = os.path.join(args.results_dir, args.exp_name)
print(args)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_label(slide_id, df):
    loc = df.index[df['slide_id']==slide_id]
    return df['label_name'][loc].item()

def check_correct(label_name, label):
    if label_name == 'Melanoma' and label == np.array([0]):
        return 1
    elif label_name == 'Nevi' and label == np.array([1]):
        return 1
    return 0


def get_score(args, model, trans_test):
    
    model = model.eval().to(device)     
    
    csv_data = pd.read_csv(args.csv_path)
    slide_ids = csv_data['slide_id']
        
    correct = 0
    df = pd.DataFrame({'slide_id':slide_ids})
    df['label_name'] = csv_data['label_name']
    df['data_split'] = csv_data['data_split']
    mel_counts, nev_counts = [], []
    for i in range(len(slide_ids)):
        
        slide_id = slide_ids[i]
        save_name = os.path.join(args.results_dir, 'score', slide_id+'.h5')
        if args.auto_skip and os.path.isfile(save_name):
            print('{} already exist in destination location, skipped'.format(slide_id))
            continue
        
        time_start = time.time()
        label_name = get_label(slide_id, csv_data)
        file_path = os.path.join(args.patch_path, slide_id+'.h5')
        
        data_set = Whole_Slide_BagV2(file_path = file_path, custom_transforms=trans_test)
        data_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False)
        scores= []
        coords = []
        class_mel = 0
        class_nev = 0
        for i, (image, coord) in tqdm(enumerate(data_loader)):
            with torch.no_grad(): 
                image = image.to(device)

                # Calculate classification loss
                logits = model(image)
                _, Y_hat = torch.max(logits.data, 1)
                class_mel += sum(Y_hat==0).item()
                class_nev += sum(Y_hat==1).item()
                scores.append(logits.detach().cpu().numpy())
                coords.append(coord.numpy())
        
        if class_mel > class_nev:
            label = np.array([0])
        else:
            label = np.array([1])


        mel_counts.append(class_mel)
        nev_counts.append(class_nev)
        correct += check_correct(label_name, label)
        

        scores = np.concatenate(scores,axis=0)
        coords = np.concatenate(coords,axis=0)
        file = h5py.File(save_name, "w")
        aset = file.create_dataset('scores', shape=scores.shape)
        aset[:] = scores
        cset = file.create_dataset('coord', shape=coords.shape)
        cset[:] = coords
        bset = file.create_dataset('pred', shape=label.shape)
        bset[:] = label
        file.close()            
                       
        time_elapsed = time.time() - time_start
            
        print('\nProcessing {} took {} s'.format(slide_id, time_elapsed))

    df['mel_con'] = mel_counts
    df['nev_con'] = nev_counts

    df['classification_result'] = df.apply(lambda x: 'Melanoma' if x['mel_con'] >
                     x['nev_con'] else 'nevi', axis=1)

    df.to_csv(os.path.join(args.classification_save_dir,"classification_"+args.exp_name+".csv"))
    print('\nSlide classification accuracy is {:.4f}'.format(correct/len(slide_ids)))
    return


if __name__ == '__main__':    
    print('loading data')

    trans_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((args.mean1, args.mean2, args.mean3), (args.std1, args.std2, args.std3)),
        ])
    
    print('loading model checkpoint')
    # Load Models
    model, input_width = instantiate_model('vgg16', True, args.num_class)
    
    os.makedirs(os.path.join(args.results_dir), exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, 'score'), exist_ok=True)
            
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    if args.model_load:
        print('loading pre-trained models')
        model.load_state_dict(torch.load(os.path.join(args.models_save_folder, args.model_load)))
        
    
    get_score(args, model, trans_test)