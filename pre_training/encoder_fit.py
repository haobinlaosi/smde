import torch
import torch.nn as nn
import os
import sklearn.model_selection
import time
from pre_training.loss import loss_CircleLoss
from pre_training.encoder import CausalCNNEncoder
import pandas as pd

class EncoderFit(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.imfs_num = args.num_imfs
        self.device = args.device
        # Initialize encoder
        if args.use_multi_gpu and len(args.device_ids)>1:
            self.encoder_t = nn.DataParallel(
                CausalCNNEncoder(args.in_channels, args.channels, args.depth, args.reduced_size, args.out_channels,
                                 args.kernel_size), device_ids=args.device_ids).to(self.device)
            self.encoder_imfs = [nn.DataParallel(
                CausalCNNEncoder(args.in_channels, args.channels, args.depth, args.reduced_size, args.imf_out_channels, args.kernel_size), device_ids=args.device_ids).to(self.device)
                for _ in range(args.num_imfs)]
        else:
            self.encoder_t = CausalCNNEncoder(args.in_channels, args.channels, args.depth, args.reduced_size,
                                              args.out_channels, args.kernel_size).to(self.device)
            self.encoder_imfs = [CausalCNNEncoder(args.in_channels, args.channels, args.depth, args.reduced_size, args.imf_out_channels,
                                                  args.kernel_size).to(self.device) for _ in range(args.num_imfs)]
        self.encoder_imfs = nn.ModuleList(self.encoder_imfs)
        # Define loss function
        self.loss = loss_CircleLoss(args.margin, args.gamma, args.lambd)
        # Define optimizer
        self.optimizer_t = torch.optim.Adam(self.encoder_t.parameters(), lr=args.lr)
        self.optimizer_tf = [torch.optim.Adam(encoder.parameters(), lr=args.lr) for encoder in self.encoder_imfs]
        self.n_epochs = 0
        self.n_iters = 0

    def fit(self, X, X_vmd, enhance_ways, noise_std, save_path, data_name, number_iters, n_epochs=None, n_iters=None, verbose=False):
        """
        param X: Time domain data
        param X_vmd: Time domain data after VMD decomposition
        param enhance_ways: Data augmentation methods
        param noise_std: standard deviation
        param save_path: save path
        param data_name: data name
        param number_iters: iterations
        """
        list_datasets = [torch.from_numpy(X) if i==0 else torch.from_numpy(X_vmd[i - 1]) for i in range(self.imfs_num+1)]

        if n_iters is None and n_epochs is None:
            n_iters = number_iters if X.size <= 100000 else number_iters
        dataset = torch.utils.data.TensorDataset(*list_datasets)
        train_generator = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        start = time.time()
        loss_log = []
        while True:
            # print("self.n_epochs:", self.n_epochs, end=' ')
            if n_epochs is not None and self.n_epochs >= n_epochs:
                break
            cum_loss = 0
            n_epoch_iters = 0
            interrupted = False
            for batch_all in train_generator:
                # print("self.n_iters:",self.n_iters, end=' ')
                if n_iters is not None and self.n_iters >= n_iters:
                    interrupted = True
                    break
                batch_all = [batch.to(self.device) for batch in batch_all]
                self.optimizer_t.zero_grad()
                for optimizer in self.optimizer_tf:
                    optimizer.zero_grad()
                # Calculate losses
                loss = self.loss(batch_all, self.encoder_t, self.encoder_imfs, enhance_ways, noise_std)
                loss.backward()
                self.optimizer_t.step()
                for optimizer in self.optimizer_tf:
                    optimizer.step()
                cum_loss += loss.item()
                n_epoch_iters += 1
                self.n_iters += 1
            if interrupted:
                break
            cum_loss /= n_epoch_iters
            loss_log.append(cum_loss)
            if verbose:
                print(f"Epoch #{self.n_epochs}: loss={cum_loss}")
            self.n_epochs += 1
        end = time.time()
        print("Time consumption for the %s iteration:%s" % (self.n_epochs, end - start), end = ' ')
        # save encoder
        self.save_encoder(os.path.join(save_path, "encoder", data_name), data_name)
        print("Encoder saved!!!", end = ' ')
        return self.encoder_t, self.encoder_imfs

    def save_encoder(self, enc_path, dataname):
        # save encoder
        if not os.path.exists(enc_path):
            os.makedirs(enc_path)
        model = {}
        for i in range(self.imfs_num+1):
            if i==0:
                model['encoder_t'] = self.encoder_t.state_dict()
            else:
                model[f'encoder_imfs{i-1}'] = self.encoder_imfs[i-1].state_dict()
        torch.save(model, os.path.join(enc_path, dataname + '_encoder.pth'))

    def load_encoder(self, enc_path, dataname):
        # load encoder
        checkpoint = torch.load(os.path.join(enc_path, dataname + f'_encoder.pth' ), map_location=lambda storage, loc: storage)
        self.encoder_t.load_state_dict(checkpoint['encoder_t'])
        for i in range(self.imfs_num):
            self.encoder_imfs[i].load_state_dict(checkpoint[f'encoder_imfs{i}'])
        return self.encoder_t, self.encoder_imfs