from sergio_simulate_data_func_celltype import SERGIO_simulate_data
import numpy as np
import os

sp_noise_params = 0
outlier_prob = 0
lib_size_effect_scale = 0
number_bins = 6
print('number_bins', number_bins)

np.random.seed(42)

for j in range(14):
        dropout_percentile = 65
        number_sc = [25,50,75,100,150,200,250,300,350,400,450,500,550,600][j]
        print('\n', f'spn{sp_noise_params}',f'out{outlier_prob}',f'lib{lib_size_effect_scale}',f'drop{dropout_percentile}','\n')
        dirname = os.path.dirname(__file__)
        adata_save_dir = os.path.join(dirname, f'data_bin{number_bins}_D{dropout_percentile}')
        if not os.path.exists(adata_save_dir):
            os.makedirs(adata_save_dir)
        SERGIO_simulate_data(adata_save_dir, dropout_percentile=dropout_percentile, noise_params=sp_noise_params, outlier_prob=outlier_prob, lib_size_effect_scale=lib_size_effect_scale, number_sc = number_sc, number_bins = number_bins, n_making_data = 10)