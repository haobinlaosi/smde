import os
import numpy as np
from vmdpy import VMD

def get_vmd_dataset(args, data_name, data, train_len):
    train_vmd_path = os.path.join(args.data_path, data_name, data_name + f"_train_vmd_imf_{args.num_imfs}.npy")
    test_vmd_path = os.path.join(args.data_path, data_name, data_name + f"_test_vmd__imf_{args.num_imfs}.npy")
    if os.path.exists(train_vmd_path) & os.path.exists(test_vmd_path):
        print("Loading VMD decomposition data!!!", end = ' ')
        train_vmd = np.load(train_vmd_path)
        test_vmd = np.load(test_vmd_path)
    else:
        print("VMD decomposing!!!",end = ' ')
        IMFs_all = []
        for j in range(data.shape[1]):
            IMFs_single = []
            for i in range(data.shape[0]):
                if np.isnan(data[i, j, :]).sum():
                    if np.where(np.isnan(data[i, j, :]))[0][-1] == len(data[i, j, :]) - 1:
                        for index, idx_n in enumerate(np.where(np.isnan(data[i, j, :]))[0][::-1]):
                            if len(data[i, j, :]) - 1 - index == idx_n:
                                t = idx_n
                    else:
                        t = len(data[i, j, :])
                else:
                    t = len(data[i, j, :])
                u, u_hat, omega = VMD(data[i, j, :t], args.alpha, args.tau, args.num_imfs, args.DC, args.init, args.tol)
                IMFs_single.append(u)
            IMFs_single = np.array(IMFs_single)
            IMFs_all.append(IMFs_single)
        IMFs_all = np.array(IMFs_all)
        IMFs_all = np.transpose(IMFs_all, (2, 1, 0, 3))
        train_vmd = IMFs_all[:, :train_len, :, :]
        test_vmd = IMFs_all[:, train_len:, :, :]
        np.save(train_vmd_path, train_vmd)
        np.save(test_vmd_path, test_vmd)
    return train_vmd, test_vmd


def get_missing_vmd_dataset(args, data, train_len):
    train_vmd_path = os.path.join(args.data_path, args.data_name, args.data_name + f"Missing_train_vmd_imf_{args.num_imfs}.npy")
    test_vmd_path = os.path.join(args.data_path, args.data_name, args.data_name + f"Missing_test_vmd_imf_{args.num_imfs}.npy")
    if os.path.exists(train_vmd_path) & os.path.exists(test_vmd_path):
        print("Loading VMD decomposition data!!!", end=' ')
        train_vmd = np.load(train_vmd_path)
        test_vmd = np.load(test_vmd_path)
    else:
        print("VMD decomposing!!!", end=' ')
        IMFs_all = []
        for j in range(data.shape[1]):
            IMFs_single = []
            for i in range(data.shape[0]):
                if np.isnan(data[i, j, :]).sum():
                    if np.where(np.isnan(data[i, j, :]))[0][-1] == len(data[i, j, :]) - 1:
                        for index, idx_n in enumerate(np.where(np.isnan(data[i, j, :]))[0][::-1]):
                            if len(data[i, j, :]) - 1 - index == idx_n:
                                t = idx_n
                    else:
                        t = len(data[i, j, :])
                else:
                    t = len(data[i, j, :])
                u, u_hat, omega = VMD(data[i, j, :t], args.alpha, args.tau, args.num_imfs, args.DC, args.init, args.tol)
                IMFs_single.append(u)
            max_dims = max(i.shape[1] for i in IMFs_single)
            IMFs_single = [np.pad(i, ((0,0), (0, max_dims-i.shape[1])), mode='constant', constant_values=0) for i in IMFs_single]
            IMFs_single = np.array(IMFs_single)
            IMFs_all.append(IMFs_single)
        IMFs_all = np.array(IMFs_all)
        IMFs_all = np.transpose(IMFs_all, (2, 1, 0, 3))
        train_vmd = IMFs_all[:, :train_len, :, :]
        test_vmd = IMFs_all[:, train_len:, :, :]
        np.save(train_vmd_path, train_vmd)
        np.save(test_vmd_path, test_vmd)
    return train_vmd, test_vmd