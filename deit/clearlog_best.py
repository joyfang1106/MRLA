# select the best acc1, acc5
import argparse
import os
import numpy as np
import pandas as pd
import csv


parser = argparse.ArgumentParser(description='clear the log and select the best')
parser.add_argument('--work-dir', default='work_dirs', type=str, 
                    help='the dir to save logs and models')
parser.add_argument('--log-dir', '-ld', default='', type=str, 
                    help='the sub dir to save logs and models of a model architecture')


def gen_log(log_dir, log_name='log.txt'):
    # log_dir = './work_dirs/ceit_tiny_patch16_224_256x4/'
    with open(os.path.join(log_dir, log_name),"r") as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]
    
    dic = eval(lines[0])
    header = list(dic.keys())
    logs = []
    for i in range(len(lines)):
        dic = eval(lines[i])
        logs.append(dic)

    # with open(os.path.join(log_dir, 'log.csv'), 'a', newline='',encoding='utf-8') as f:
    with open(os.path.join(log_dir, 'log.csv'), 'w', newline='',encoding='utf-8') as f:
        # 'a': additional
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(logs)
        f.close()

              
def main():
    global args
    args = parser.parse_args()
    log_dir = "%s/%s/"%(args.work_dir, args.log_dir)
    gen_log(log_dir)
    
    log_df = pd.read_csv(os.path.join(log_dir, 'log.csv'))
    log_df[['train_lr', 'epoch']] = log_df[['epoch', 'train_lr']]
    # log_df.head()
    log_df = log_df.rename(columns={'train_lr':'epoch', 'epoch':'train_lr'})
    log_df.to_csv(os.path.join(log_dir, 'log.csv'), index=False, sep=',')
    
    # select the best top1 top5
    acc1_list = log_df['test_acc1']
    acc1_list = acc1_list.to_list()
    best_acc1 = np.max(acc1_list)
    best_idx1 = acc1_list.index(np.max(acc1_list))
    best_acc1_dic = log_df.iloc[best_idx1].to_dict()
    
    acc5_list = log_df['test_acc5']
    acc5_list = acc5_list.to_list()
    best_acc5 = np.max(acc5_list)
    best_idx5 = acc5_list.index(np.max(acc5_list))
    best_acc5_dic = log_df.iloc[best_idx5].to_dict()
    
    with open(os.path.join(log_dir, 'best.txt'), "w") as f:
        f.write("* best Top-1 at epoch {}: Acc@1: {:.3f}, Acc@5: {:.3f}, Err@1: {:.3f}, Err@5: {:.3f} \n".format(int(best_acc1_dic['epoch']), best_acc1_dic['test_acc1'], best_acc1_dic['test_acc5'], 100-best_acc1_dic['test_acc1'], 100-best_acc1_dic['test_acc5']))
        f.write("* best Top-5 at epoch {}: Acc@1: {:.3f}, Acc@5: {:.3f}, Err@1: {:.3f}, Err@5: {:.3f} \n".format(int(best_acc5_dic['epoch']), best_acc5_dic['test_acc1'], best_acc5_dic['test_acc5'], 100-best_acc5_dic['test_acc1'], 100-best_acc5_dic['test_acc5']))
        f.close()
    
    print ("-" * 80)
    print ("* best Top-1 at epoch {}: Acc@1: {:.3f}, Acc@5: {:.3f}, Err@1: {:.3f}, Err@5: {:.3f}".format(int(best_acc1_dic['epoch']), best_acc1_dic['test_acc1'], best_acc1_dic['test_acc5'], 100-best_acc1_dic['test_acc1'], 100-best_acc1_dic['test_acc5']))
    print ("-" * 80)
    print ("* best Top-5 at epoch {}: Acc@1: {:.3f}, Acc@5: {:.3f}, Err@1: {:.3f}, Err@5: {:.3f}".format(int(best_acc5_dic['epoch']), best_acc5_dic['test_acc1'], best_acc5_dic['test_acc5'], 100-best_acc5_dic['test_acc1'], 100-best_acc5_dic['test_acc5']))
    print ("-" * 80)
    
    
    
if __name__ == '__main__':
    main()