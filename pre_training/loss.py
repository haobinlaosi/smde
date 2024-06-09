import torch.nn as nn
import torch
from pre_training.data_enhance import fun_times_data_enhance, fun_imfs_data_enhance, subsequence_enhance, flip_enhance

class loss_CircleLoss(nn.modules.loss._Loss):
    def __init__(self, margin, gamma, lambd):
        super(loss_CircleLoss, self).__init__()
        self.margin = margin
        self.gamma = gamma
        self.lambd = lambd
        self.delta_p = 1 - margin
        self.delta_n = margin
        self.soft_plus = nn.Softplus()

    def forward(self, batch, encoder_t, encoder_imfs, enhance_ways, noise_std):
        # weight coefficient 
        ratio = torch.full((len(encoder_imfs) + 1, 1), self.lambd)
        ratio[0] = 1.0
        l_b = batch[0].size(0)

        logit_p_list, logit_n_list = [], []
        for i in range(len(encoder_imfs) + 1):
            if i == 0:
                if enhance_ways=="Gaussian":
                    pre_t = encoder_t(torch.cat([batch[0].float(), fun_times_data_enhance(batch[0], noise_std).float()], dim=0)) # 2B*outchannels
                elif enhance_ways == "Subsequence":
                    sub_query, sub_pn = subsequence_enhance(batch[0])
                    pre_t = torch.cat((encoder_t(sub_query), encoder_t(sub_pn)), dim=0).float()
                elif enhance_ways == "Flip":
                    pre_t = encoder_t(torch.cat([batch[0].float(), flip_enhance(batch[0]).float()], dim=0))
                else:
                    raise ValueError("Parameter enhance_ways input error")

                sim_matrix_t = nn.functional.cosine_similarity(pre_t.unsqueeze(1), pre_t.unsqueeze(0), dim=2)
                sim_mat_t = torch.tril(sim_matrix_t, diagonal=-1)[:, :-1]  # 2B x (2B-1)
                sim_mat_t += torch.triu(sim_matrix_t, diagonal=1)[:, 1:]
                sim_mat_t = sim_mat_t[:l_b, :]
                sim_p = torch.tensor([sim_mat_t[i - l_b+1, i] for i in range(l_b-1, 2 * l_b-1)])
                sim_n = sim_mat_t[:, :l_b - 1]

            else:
                if enhance_ways=="Gaussian":
                    pre_imf = encoder_imfs[i-1](torch.cat([batch[i].float(), fun_times_data_enhance(batch[i], noise_std).float()], dim=0))
                elif enhance_ways == "Subsequence":
                    sub_query, sub_pn = subsequence_enhance(batch[i])
                    pre_imf = torch.cat((encoder_imfs[i-1](sub_query), encoder_imfs[i-1](sub_pn)), dim=0).float()
                elif enhance_ways == "Flip":
                    pre_imf = encoder_imfs[i-1](torch.cat([batch[i].float(), flip_enhance(batch[i]).float()], dim=0))
                else:
                    raise ValueError("Parameter enhance_ways input error")

                sim_matrix_imf = nn.functional.cosine_similarity(pre_imf.unsqueeze(1), pre_imf.unsqueeze(0), dim=2)
                sim_mat_imf = torch.tril(sim_matrix_imf, diagonal=-1)[:, :-1]  # 2B x (2B-1)
                sim_mat_imf += torch.triu(sim_matrix_imf, diagonal=1)[:, 1:]
                sim_mat_imf = sim_mat_imf[:l_b, :]
                sim_p = torch.tensor([sim_mat_imf[i - l_b+1, i] for i in range(l_b-1, 2 * l_b-1)])
                sim_n = sim_mat_imf[:, :l_b - 1]

            ap = torch.clamp_min(- sim_p.detach() + 1 + self.margin, min=0.)
            logit_p_single = ap * (sim_p - self.delta_p) * self.gamma
            logit_p_list.append(logit_p_single)

            an = torch.clamp_min(sim_n.detach() + self.margin, min=0.)
            logit_n_single = an * (sim_n - self.delta_n) * self.gamma
            logit_n_list.append(logit_n_single)

        logit_p = torch.stack(logit_p_list, dim=0) + torch.log(ratio)

        ratio_list = [ratio[i, :] for i in range(len(encoder_imfs) + 1)]
        logit_n = [x.cpu() + torch.log(y) for x, y in zip(logit_n_list, ratio_list)]
        logit_n = torch.cat(logit_n, dim=1)
        loss = torch.mean(self.soft_plus(torch.logsumexp(logit_n, dim=1) - torch.logsumexp(logit_p, dim=0)))
        return loss