import os
import cv2
import time
import argparse
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from datasets.dataset_h5 import Dataset_All_Bags, h5_to_patch_new

# Start from ./patches_all_norm folder
# Based on annotations_new to check if certain patch is in region or not
# Based on annotations_other to check if it goes into other folder
# Keep only in-region patches for training this model
# Assign them to corresponding train,val,test folder


def log_check_file(logfname):
    """
    Count the number of patches for each WSI image.
    """
    lineList = [line.rstrip('\n').split(' ')[0] for line in open(logfname)]
    return lineList

def log_patches_count(logfname, filename, count):
    """
    Count the number of patches for each WSI image.
    """
    f = open(logfname, 'a+')
    f.write(filename+' '+str(count)+'\n')
    f.close()

def make_dirs(args):
    os.makedirs(args.feat_dir, exist_ok=True)
    for name in ["train", "val", "test"]:
        os.makedirs(os.path.join(args.feat_dir,name), exist_ok=True)
        for label in ["Melanoma", "Nevi", "Other"]:
            os.makedirs(os.path.join(args.feat_dir,name, label), exist_ok=True) 
    return

def checkinout(patch_mask, annotation_ratio=1.00):
    """
    Checks whether or not an image chip should be saved
    :param mask: array of pixel in mask
    :return: bool
    """
    if np.sum(patch_mask>0) / patch_mask.size >= float(annotation_ratio):
        save = True
    else:
        save = False
    return save
          

def makemask(w, h, xml_path):
    """
    Reads xml file and makes annotation mask for entire slide image
    :param annotation_key: name of the annotation key file
    :param size: size of the whole slide image
    :param xml_path: path to the xml file
    :return: annotation mask
    :return: dictionary of annotation keys and color codes
    """
    # Import xml file and get root
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Generate annotation array and key dictionary
    mat = np.zeros((h, w), dtype='uint8') # should use mat = np.zeros((h, w), dtype='uint8'), since cv2.fillPoly dim range is like that
    contours = []
    for reg in root.iter('Region'):
        points = []
        for child in reg.iter('Vertices'):
            for vert in child.iter('Vertex'):
                x = int(round(float(vert.get('X'))))
                y = int(round(float(vert.get('Y'))))
                points.append((x, y))
        cnt = np.array(points).reshape((-1, 1, 2)).astype(np.int32)
        cv2.fillPoly(mat, [cnt], 255)
        contours.append(cnt)
    return mat


def compute_w_loader(args, file_path, xml_path, output_path, bag_name, logfname, patch_size=256, target_patch_size=-1):
    """
    args:
        file_path: directory of bag (.h5 file)
        output_path: directory to save extracted patches
        model: pytorch model
        batch_size: batch_size for computing features in batches
        verbose: level of feedback
    """
    dataset = h5_to_patch_new(file_path=file_path, target_patch_size=target_patch_size)
    w, h = dataset.level_dim    
    n = len(dataset)
    count_patch = 0
    mat = makemask(w, h, xml_path)
    for i in range(n):
        x, y = dataset[i]
        img_mask = mat[y[1]:(y[1] + patch_size),y[0]:y[0] + patch_size]
        inregion = checkinout(img_mask)
        if inregion:
            count_patch += 1
            save_path = '_'.join([output_path,str(y[0]),str(y[1])])+'.png'
            x.save(save_path)
    log_patches_count(logfname, bag_name.rstrip(".h5"), count_patch)
    return


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_dir', type=str)
parser.add_argument('--csv_path', type=str)
parser.add_argument('--xml_annotation_new', type=str)
parser.add_argument('--xml_annotation_other', type=str)
parser.add_argument('--feat_dir', type=str)
parser.add_argument('--auto_skip', default=False, action='store_true')
parser.add_argument('--patch_size', type=int, default=256)
parser.add_argument('--target_patch_size', type=int, default=-1,
                    help='the desired size of patches for optional scaling before feature embedding')
args = parser.parse_args()


if __name__ == '__main__':

    print('initializing dataset')
    csv_path = args.csv_path
    df = pd.read_csv(csv_path)
    bags_dataset = Dataset_All_Bags(args.data_dir, csv_path)
    logfname = os.path.join(args.feat_dir,'patch_count.txt')
    logfname_other = os.path.join(args.feat_dir,'patch_count_other.txt')
    
    if args.auto_skip:
        processed_list = log_check_file(logfname)        
    else:
        processed_list = []
        
    make_dirs(args)

    total = len(bags_dataset)
    
    for bag_candidate_idx in range(total):
        bag_candidate = bags_dataset[bag_candidate_idx]
        bag_name = os.path.basename(os.path.normpath(bag_candidate))
        if args.auto_skip and bag_name.rstrip(".h5") in processed_list:
            print('skipped {}'.format(bag_name))
            continue
            
        target_folder = os.path.join(args.feat_dir,df['data_split'][bag_candidate_idx], df['label_name'][bag_candidate_idx])
        if '.h5' in bag_candidate:

            print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
            print(bag_name)
            xml_path = os.path.join(args.xml_annotation_new, bag_name.split('.')[0]+'.xml')
            output_path = os.path.join(target_folder, bag_name.split('.')[0])
            file_path = bag_candidate
            time_start = time.time()
            compute_w_loader(args, file_path, xml_path, output_path, bag_name, logfname,
                                                target_patch_size=args.target_patch_size)
            time_elapsed = time.time() - time_start
            print('\nsplitting patches for {} took {} s'.format(bag_name, time_elapsed))
            
        contour_xml_path = os.path.join(args.xml_annotation_other, bag_name.split('.')[0]+'.xml')
        if os.path.isfile(contour_xml_path):
            print('\nprocessing {} to extract other patches'.format(bag_name))
            other_target_folder = os.path.join(args.feat_dir,df['data_split'][bag_candidate_idx], 'Other')
            output_path = os.path.join(other_target_folder, bag_name.split('.')[0])
            file_path = bag_candidate
            time_start = time.time()
            compute_w_loader(args, file_path, contour_xml_path, output_path, bag_name, logfname_other,
                                                target_patch_size=args.target_patch_size)
            time_elapsed = time.time() - time_start
            print('\nsplitting patches for {} took {} s'.format(bag_name, time_elapsed))    