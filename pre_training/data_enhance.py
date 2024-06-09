import torch
import numpy as np


def fun_imfs_data_enhance(x, noise_std):
    return x + torch.normal(mean=0., std=noise_std, size=x.shape, device=x.device)

def fun_times_data_enhance(x, noise_std):
    return x + torch.normal(mean=0., std=noise_std, size=x.shape, device=x.device)

def subsequence_enhance(x):
    batch_size = x.size(0)
    length = x.size(2)
    # Choice of length of positive and negative samples
    length_pos_neg = np.random.randint(1, high=length + 1)
    # We choose for each batch example a random interval in the time series, which is the 'anchor'
    random_length = np.random.randint(length_pos_neg, high=length + 1)  # Length of anchors
    beginning_batches = np.random.randint(0, high=length - random_length + 1, size=batch_size)  # Start of anchors
    # The positive samples are chosen at random in the chosen anchors
    beginning_samples_pos = np.random.randint(0, high=random_length - length_pos_neg + 1,
                                                 size=batch_size)  # Start of positive samples in the anchors
    # Start of positive samples in the batch examples
    beginning_positive = beginning_batches + beginning_samples_pos
    # End of positive samples in the batch examples
    end_positive = beginning_positive + length_pos_neg

    query = torch.cat([x[j: j + 1, :, beginning_batches[j]: beginning_batches[j] + random_length] for j in range(batch_size)]).to(torch.float32)
    positive_negative = torch.cat([x[j: j + 1, :, end_positive[j] - length_pos_neg: end_positive[j]] for j in range(batch_size)]).to(torch.float32)
    return query, positive_negative

def flip_enhance(x):
    reversed_tensor = torch.flip(x, dims=[2])
    return reversed_tensor