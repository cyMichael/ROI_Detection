import os
import argparse
import pandas as pd


parser = argparse.ArgumentParser(description='IOU Analysis script')
parser.add_argument('--results_dir', type=str,
                        help='folder in which to save model')
parser.add_argument('--csv_dir', type=str,
                        help='folder in which to save model')
args = parser.parse_args()


def log_check_file(logfname, num_highlight=None):
    """
    Count the number of patches for each WSI image.
    """
    slide_ids = [line.rstrip('\n').split(' ')[0] for line in open(logfname)]
    num_patches = [int(line.rstrip('\n').split(' ')[1]) for line in open(logfname)]
    num_inregion_highlight = [int(line.rstrip('\n').split(' ')[2]) for line in open(logfname)]
    num_inregion = [int(line.rstrip('\n').split(' ')[3]) for line in open(logfname)]
    if num_highlight is None:
        num_highlight = [int(line.rstrip('\n').split(' ')[4]) for line in open(logfname)]        
    ratio, zero_list = get_ratio(num_inregion_highlight, num_inregion, num_highlight, slide_ids)
    return slide_ids, num_patches, num_inregion_highlight, num_inregion, num_highlight, ratio, zero_list

def get_ratio(num_inregion_highlight, num_inregion, num_highlight_global, slide_ids):
    ratio = []
    zero_list = []
    for i in range(len(num_inregion)):
        if num_inregion[i] != 0:
            num_union = num_inregion[i] + num_highlight_global[i]-num_inregion_highlight[i]
            ratio.append(num_inregion_highlight[i]/num_union)
        else:
            zero_list.append(slide_ids[i])
            ratio.append(0)                        
    return ratio, zero_list


def remove_pgem(slide_ids):
    slide_num = []
    for i in range(len(slide_ids)):
        slide_num.append(slide_ids[i][4:])
    return slide_num


def get_result_df(name, i, df_label, num_highlight=None):   
    logfname = os.path.join(args.results_dir, name+'.txt')
    slide_ids, num_patches, num_inregion_highlight, num_inregion, num_highlight, ratio, zero_list = log_check_file(logfname, num_highlight)

    sum(ratio)/(len(ratio)-len(zero_list))
    
    df = pd.DataFrame({'slide_id':slide_ids})
    if i == 0:
        df['slide_num'] = remove_pgem(slide_ids)
        df['num_patches'] = num_patches
        df['num_inregion'] = num_inregion
        df = pd.merge(df, df_label, how='left', on='slide_id')
    df['ratio_'+name] = ratio
    
    percent = sum(ratio)/(len(ratio)-len(zero_list))
    print("average ratio is: {}".format(percent))
    return df, num_highlight

df_label = pd.read_csv(os.path.join(args.csv_dir,'melanomal.csv'))
save_name = 'summary_iou_final.csv'
exp_name_list = ['pcla_3class']

dfs = []
for i in range(len(exp_name_list)):
    if i == 0:
        tmp_df, num_highlight = get_result_df(exp_name_list[i],i, df_label)
    else:
        tmp_df, _ = get_result_df(exp_name_list[i],i, df_label, num_highlight)
    dfs.append(tmp_df)

df_main = dfs[0]

if len(dfs) >= 2:
    for i in range(1,len(dfs)):
        df_main = pd.merge(df_main, dfs[i], how='left', on='slide_id')


df_main.to_csv(os.path.join(args.results_dir,save_name))