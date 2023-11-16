import torch
import numpy as np

class DeepKINETDataSet(torch.utils.data.Dataset):
    def __init__(self, s, u, norm_mat, norm_mat_u):
        self.s = s
        self.u = u
        self.norm_mat = norm_mat
        self.norm_mat_u = norm_mat_u
    def __len__(self):
        return(self.s.shape[0])
    def __getitem__(self, idx):
        idx_s = self.s[idx]
        idx_u = self.u[idx]
        idx_norm_mat = self.norm_mat[idx]
        idx_norm_mat_u = self.norm_mat_u[idx]
        return(idx_s, idx_u, idx_norm_mat, idx_norm_mat_u)


class DeepKINETDataManager():
    def __init__(self, s, u, test_ratio, batch_size, num_workers, validation_ratio):
        snorm_mat = torch.mean(s, dim=1, keepdim=True) * torch.mean(s, dim=0, keepdim=True)
        snorm_mat =  torch.mean(s) * snorm_mat / torch.mean(snorm_mat)
        unorm_mat = torch.mean(u, dim=1, keepdim=True) * torch.mean(u, dim=0, keepdim=True)
        unorm_mat =  torch.mean(u) * unorm_mat / torch.mean(unorm_mat)
        self.s = s
        self.u = u
        self.norm_mat = snorm_mat
        self.norm_mat_u = unorm_mat
        total_num = s.shape[0]
        validation_num = int(total_num * validation_ratio)
        test_num = int(total_num * test_ratio)
        np.random.seed(42)
        idx = np.random.permutation(np.arange(total_num))
        validation_idx, test_idx, train_idx = idx[:validation_num], idx[validation_num:(validation_num +  test_num)], idx[(validation_num +  test_num):]
        self.validation_idx, self.test_idx, self.train_idx = validation_idx, test_idx, train_idx
        self.validation_s = s[validation_idx]
        self.validation_u = u[validation_idx]
        self.validation_norm_mat = self.norm_mat[validation_idx]
        self.validation_norm_mat_u = self.norm_mat_u[validation_idx]
        self.test_s = s[test_idx]
        self.test_u = u[test_idx]
        self.test_norm_mat = self.norm_mat[test_idx]
        self.test_norm_mat_u = self.norm_mat_u[test_idx]
        self.train_eds = DeepKINETDataSet(self.s[train_idx], self.u[train_idx], self.norm_mat[train_idx], self.norm_mat_u[train_idx])
        self.train_loader = torch.utils.data.DataLoader(
            self.train_eds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=True)