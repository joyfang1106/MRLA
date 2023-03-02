# select the best acc1, acc5
import argparse
import os
import numpy as np
import pandas as pd


parser = argparse.ArgumentParser(description='Clear the log and select the best')
parser.add_argument('--work-dir', default='work_dirs', type=str, 
                    help='the dir to save logs and models')
parser.add_argument('--log-dir', '-ld', default='resnet50_mrlal_', type=str, 
                    help='the sub dir to save logs and models of a model architecture')


def clear_log(lines):
    '''
    extract two lists of epochs and accuracies
    '''
    acc_list = []
    epo_list = []
    for i in range(len(lines)):
        epo_i = lines[i].split(' ')[0]
        acc_i = lines[i].split('(')[1]
        acc_i = acc_i.split(',')[0]
        epo_list.append(int(epo_i))
        acc_list.append(float(acc_i))
    return acc_list, epo_list

def read_acclog(log_dir, log_name):
    with open(os.path.join(log_dir, log_name),"r") as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]
    
    acc_list, epo_list = clear_log(lines) 
    # acc_list = [format(acc, '.4f') for acc in acc_list]

    return acc_list, epo_list

def read_losslog(log_dir, log_name):
    with open(os.path.join(log_dir, log_name),"r") as f:
        lines = f.readlines()
    epo_list = [x.split(' ')[0] for x in lines]
    epo_list = [int(epo) for epo in epo_list]
    loss_list = [x.split(' ')[1] for x in lines]
    loss_list = [loss.replace('\n', '') for loss in loss_list]
    # loss_list = [format(float(loss), '.4f') for loss in loss_list]
    loss_list = [float(loss) for loss in loss_list]
    return loss_list, epo_list

def generate_log(log_dir):
    '''
    save all logs into .txt and .csv files
    '''
    train_acc1, epo_list = read_acclog(log_dir, log_name='train_acc1.txt')
    train_acc5, _ = read_acclog(log_dir, log_name='train_acc5.txt')
    val_acc1, _ = read_acclog(log_dir, log_name='val_acc1.txt')
    val_acc5, _ = read_acclog(log_dir, log_name='val_acc5.txt')
    train_loss, _ = read_losslog(log_dir, log_name='loss_plot.txt')
    
    df = pd.DataFrame(columns=['epoch', 'train_acc1', 'train_acc5', 
                               'val_acc1', 'val_acc5', 'train_loss'])
    df['epoch'] = epo_list
    df['train_acc1'] = train_acc1
    df['train_acc5'] = train_acc5
    df['val_acc1'] = val_acc1
    df['val_acc5'] = val_acc5
    df['train_loss'] = train_loss
    
    if os.path.exists(os.path.join(log_dir, 'valloss_plot.txt')):
        val_loss, _ = read_losslog(log_dir, log_name='valloss_plot.txt')
        df['val_loss'] = val_loss
    df.to_csv(os.path.join(log_dir, 'log.csv'), index=False, sep=',')
    
    with open(os.path.join(log_dir, 'log.txt'), "w") as f:
        for i in range(len(epo_list)):
            if os.path.exists(os.path.join(log_dir, 'valloss_plot.txt')):
                log_i = '{} train_Acc@1 {:.3f} train_Acc@5 {:.3f} Acc@1 {:.3f} Acc@5 {:.3f} train_Loss {:.4f} Loss {:.4f} \n'.format(epo_list[i], train_acc1[i], train_acc5[i], val_acc1[i], val_acc5[i], train_loss[i], val_loss[i])
            else:
                log_i = '{} train_Acc@1 {:.3f} train_Acc@5 {:.3f} Acc@1 {:.3f} Acc@5 {:.3f} train_Loss {:.4f} \n'.format(epo_list[i], train_acc1[i], train_acc5[i], val_acc1[i], val_acc5[i], train_loss[i])
                
            f.write(log_i)
            
        
               
def main():
    global args
    args = parser.parse_args()
    log_dir = "%s/%s/"%(args.work_dir, args.log_dir)
    generate_log(log_dir)
    
    log_df = pd.read_csv(os.path.join(log_dir, 'log.csv'))  
    
    # select the best top1 top5
    acc1_list = log_df['val_acc1']
    acc1_list = acc1_list.to_list()
    best_acc1 = np.max(acc1_list)
    best_idx1 = acc1_list.index(np.max(acc1_list))
    best_acc1_dic = log_df.iloc[best_idx1].to_dict()
    
    acc5_list = log_df['val_acc5']
    acc5_list = acc5_list.to_list()
    best_acc5 = np.max(acc5_list)
    best_idx5 = acc5_list.index(np.max(acc5_list))
    best_acc5_dic = log_df.iloc[best_idx5].to_dict()
    
    with open(os.path.join(log_dir, 'best.txt'), "w") as f:
        f.write("* best Top-1 at epoch {}: Acc@1: {:.3f}, Acc@5: {:.3f}, Err@1: {:.3f}, Err@5: {:.3f} \n".format(int(best_acc1_dic['epoch']), best_acc1_dic['val_acc1'], best_acc1_dic['val_acc5'], 100-best_acc1_dic['val_acc1'], 100-best_acc1_dic['val_acc5']))
        f.write("* best Top-5 at epoch {}: Acc@1: {:.3f}, Acc@5: {:.3f}, Err@1: {:.3f}, Err@5: {:.3f} \n".format(int(best_acc5_dic['epoch']), best_acc5_dic['val_acc1'], best_acc5_dic['val_acc5'], 100-best_acc5_dic['val_acc1'], 100-best_acc5_dic['val_acc5']))
        f.close()
    
    print ("-" * 80)
    print ("* best Top-1 at epoch {}: Acc@1: {:.3f}, Acc@5: {:.3f}, Err@1: {:.3f}, Err@5: {:.3f}".format(int(best_acc1_dic['epoch']), best_acc1_dic['val_acc1'], best_acc1_dic['val_acc5'], 100-best_acc1_dic['val_acc1'], 100-best_acc1_dic['val_acc5']))
    print ("-" * 80)
    print ("* best Top-5 at epoch {}: Acc@1: {:.3f}, Acc@5: {:.3f}, Err@1: {:.3f}, Err@5: {:.3f}".format(int(best_acc5_dic['epoch']), best_acc5_dic['val_acc1'], best_acc5_dic['val_acc5'], 100-best_acc5_dic['val_acc1'], 100-best_acc5_dic['val_acc5']))
    print ("-" * 80)
    
    
    
if __name__ == '__main__':
    main()
    
    