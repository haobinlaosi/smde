import os
from get_data.load_data import load_ucr_dataset, load_uea_dataset
from get_data.load_vmd_data import get_vmd_dataset, get_missing_vmd_dataset
import numpy as np
import glob
import datetime
from pre_training.encoder_fit import EncoderFit
import pandas as pd
from downstream_task.model import Downstream_Classifier
import argparse
import torch
import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='Training parameters')

parser.add_argument('--data_path', type=str, default='/Data/comlearning/algorithm_test/SMDE/data', help='path of data')
parser.add_argument('--data_folder', type=str, default='UCR', help='What data to use:UCR or UEA')
parser.add_argument('--save_path', type=str, default='/Data/comlearning/algorithm_test/algorithm_2/results', help='save path')
# VMD parameters
parser.add_argument('--num_imfs', type=int, default=3, help='The number of IMF components')
parser.add_argument('--alpha', type=int, default=2000, help='The alpha parameter controls the smoothness or sparsity.')
parser.add_argument('--tau', type=float, default=0., help='The tau parameter controls the strength of regularization.')
parser.add_argument('--DC', type=int, default=0, help='The DC parameter determines whether to remove the DC component.')
parser.add_argument('--init', type=int, default=1, help='The init parameter determines the initialization method.')
parser.add_argument('--tol', type=float, default=1e-7, help='The tol parameter controls the tolerance for convergence.')
# encoder parameters
parser.add_argument('--in_channels', type=int, default=1, help='input signal')
parser.add_argument('--channels', type=int, default=40, help='hidden signal')
parser.add_argument('--reduced_size', type=int, default=160, help='output signal')
parser.add_argument('--out_channels', type=int, default=120, help='encoding output')
parser.add_argument('--depth', type=int, default=10, help='depth of convolution')
parser.add_argument('--kernel_size', type=int, default=3, help='kernel size')
parser.add_argument('--batch_size', type=int, default=10, help='The size of the batch')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--n_iters', type=int, default=1500, help='Iterations')
# loss parameters
parser.add_argument('--enhance_ways', type=str, default='Gaussian', help='Data augmentation methods:Gaussian、Subsequence or Flip')
parser.add_argument('--noise_std', type=float, default=0.05, help='Standard deviation of Gaussian noise')
parser.add_argument('--gamma', type=int, default=80, help='scaling factor')
parser.add_argument('--margin', type=float, default=0.25, help='The margin parameter controls the separation between positive and negative samples.')

parser.add_argument('--use_multi_gpu', default=False, help='Whether to use multiple GPUs')
parser.add_argument('--device_ids', default=[0,1,2,3], help='GPU number')
args = parser.parse_args()

args.imf_out_channels = args.out_channels//args.num_imfs
args.weight_coefficient = 1/args.num_imfs
args.lambd = args.weight_coefficient/args.num_imfs

def main(dataname):
    if torch.cuda.is_available():
        print("cuda:" + str(args.device_ids[0]))
        args.device = torch.device("cuda:" + str(args.device_ids[0]))
    else:
        args.device = torch.device('cpu')

    print("loading data!!!", end=' ')
    train, train_labels, test, test_labels = load_ucr_dataset(args.data_path, dataname) if args.data_folder == 'UCR' else load_uea_dataset(args.data_path, dataname)
    args.in_channels = train.shape[1]
    print("train, train_labels, test, test_labels:", train.shape, train_labels.shape, test.shape, test_labels.shape)
    # Determine if there are missing values in the data
    if np.isnan(train).any() or np.isnan(test).any():
        print(dataname + ':There are missing values present！')
        train_vmd, test_vmd = get_missing_vmd_dataset(args, dataname, np.concatenate((train, test), axis=0), train.shape[0])
        train[np.isnan(train)] = 0
        test[np.isnan(test)] = 0
    else:
        train_vmd, test_vmd = get_vmd_dataset(args, dataname, np.concatenate((train, test), axis=0), train.shape[0])
    print("train_vmd, test_vmd:", train_vmd.shape, test_vmd.shape)

    # Pre training
    print("Pre Training!!!", end = ' ')
    ## Initialize encoder
    model_encoder = EncoderFit(args)
    encoder_path = os.path.join(args.save_path, 'encoder', dataname) # encoder saved path
    ## If the encoder has already been trained, load it; otherwise, train the encoder
    if os.path.exists(encoder_path):
        print("loading encoder")
        encoder_t, encoder_imfs = model_encoder.load_encoder(encoder_path, dataname)
    else:
        print("training encoder")
        encoder_t, encoder_imfs = model_encoder.fit(train, train_vmd, args.enhance_ways, args.noise_std, args.save_path, dataname, args.n_iters)

    # Downstream tasks
    print("Downstream Tasks!!!")
    classifier_path = os.path.join(args.save_path, 'classifier') # Classifier save path
    num_classes = np.shape(np.unique(train_labels, return_counts=True)[1])[0]
    ## Initialize classifier
    down_classifier = Downstream_Classifier(args.out_channels, args.imf_out_channels, encoder_t, encoder_imfs, num_classes, args.device)
    ## classifier training
    down_classifier.fit(train, train_vmd, train_labels, classifier_path, dataname)
    ## classifier testting
    acc, f_score, precision, recall = down_classifier.test(test, test_vmd, test_labels)
    print(f"acc:{acc}, f_score:{f_score}, precision:{precision}, recall:{recall}")
    return train.shape, test.shape, num_classes, acc, f_score, precision, recall

if __name__ == '__main__':
    # data_folder = 'UEA'
    if args.data_folder == 'UCR':
        args.data_path = os.path.join(args.data_path, args.data_folder, 'UCRArchive_2018')
    elif args.data_folder == 'UEA':
        args.data_path = os.path.join(args.data_path, args.data_folder, 'Multivariate_arff')

    args.save_path = os.path.join(args.save_path, args.data_folder, datetime.datetime.now().strftime("%Y%m%d"))

    data_name = sorted(os.listdir(args.data_path), key=lambda x: x.lower())[:3]
    results = []
    for index, dataname in enumerate(data_name):
        print(f'{index}    {dataname}')
        data_res = {}
        data_res['dataname'] = dataname
        train_size, test_size, num_y, acc, f_score, precision, recall = main(dataname)
        data_res['train_size'] = train_size
        data_res['test_size'] = test_size
        data_res['classes'] = num_y
        data_res['acc'] = acc
        data_res['f_score'] = f_score
        data_res['precision'] = precision
        data_res['recall'] = recall
        results.append(data_res.copy())
    results = pd.DataFrame(results, columns=list(results[0].keys()))
    print(results)
    results.to_excel(
        os.path.join(
            args.save_path,
            f'SMDE-data({args.data_folder})({datetime.datetime.now().strftime("%Y%m%d%H%M%S")}).xlsx'
        )
        , index=True
    )
    print("Successfully saved results！！！")