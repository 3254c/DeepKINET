#This code is cited from Dibaeinia P, Sinha S. SERGIO: A Single-Cell Expression Simulator Guided by Gene Regulatory Networks. Cell Syst. 2020.

import os
import anndata as ad
import numpy as np
import pandas as pd
from sergio_celltype import sergio

def SERGIO_simulate_data(adata_save_dir, dropout_percentile=0, noise_params=0, outlier_prob=0, lib_size_effect_scale=0, number_sc = 300, number_bins = 6, n_making_data = 10):        
        for n in range(n_making_data):
            number_genes = 100
            decays=0.8
            splice_ratio = 4
            gamma=decays/splice_ratio


            genes_base_decays = np.random.uniform(0.5, 1.5, number_genes) * decays
            genes_base_gamma = np.random.uniform(0.5, 1.5, number_genes) * gamma

            bin_amp_ratio_beta = np.random.uniform(0.75, 1.25, (number_bins, number_genes))
            bin_amp_ratio_gamma = np.random.uniform(0.75, 1.25, (number_bins, number_genes))

            SERGIO_beta = bin_amp_ratio_beta * genes_base_decays
            SERGIO_gamma = bin_amp_ratio_gamma * genes_base_gamma
            SERGIO_splice_ratio = SERGIO_beta / SERGIO_gamma

            dirname = os.path.dirname(__file__)
            if number_bins == 6:
                df = pd.read_csv(os.path.join(dirname, 'bMat_cID7.tab'), sep='\t', header=None, index_col=None)
            elif number_bins == 7:
                df = pd.read_csv(os.path.join(dirname, 'bMat_cID11.tab'), sep='\t', header=None, index_col=None)
            else:
                print('number_bins should be 6 or 7')
                return
            bMat = df.values
            sim = sergio(number_genes=number_genes, number_bins = number_bins, number_sc = number_sc, 
                        noise_params = noise_params, decays=SERGIO_beta, 
                        splice_ratio = SERGIO_splice_ratio, sampling_state = 1, 
                        noise_params_splice = noise_params, noise_type='dpd', bifurcation_matrix= bMat)
            if number_bins == 6:
                sim.build_graph(input_file_taregts = os.path.join(dirname, 'Interaction_cID_7.txt'), input_file_regs= os.path.join(dirname, 'Regs_cID_7.txt'), shared_coop_state=2)
            elif number_bins == 7:
                sim.build_graph(input_file_taregts = os.path.join(dirname, 'Interaction_cID_11.txt'), input_file_regs=os.path.join(dirname, 'Regs_cID_11.txt'), shared_coop_state=2)

            sim.simulate_dynamics()
            exprU, exprS = sim.getExpressions_dynamics()

            clean_count_matrix_U, clean_count_matrix_S = sim.convert_to_UMIcounts_dynamics(exprU, exprS)
            clean_count_matrix_U = np.concatenate(clean_count_matrix_U, axis = 1)
            clean_count_matrix_S = np.concatenate(clean_count_matrix_S, axis = 1)
            clean_count_matrix_total = clean_count_matrix_S + clean_count_matrix_U


            if outlier_prob!=0:
                print('outlier effect')
                exprU_n, exprS_n = sim.outlier_effect_dynamics(exprU, exprS, outlier_prob = outlier_prob, mean = 0.8, scale = 1)
            if lib_size_effect_scale!=0:
                print('lib_size_effect')
                libFactor, exprU_n, exprS_n = sim.lib_size_effect_dynamics(exprU, exprS, mean = 0, scale = lib_size_effect_scale)
            if dropout_percentile!=0:
                print('dropout_effect')
                binary_indU, binary_indS = sim.dropout_indicator_dynamics(exprU, exprS, percentile = dropout_percentile)
                exprU_n = np.multiply(binary_indU, exprU)
                exprS_n = np.multiply(binary_indS, exprS)
            if outlier_prob!=0 or lib_size_effect_scale!=0 or dropout_percentile!=0:
                count_matrix_U_D, count_matrix_S_D = sim.convert_to_UMIcounts_dynamics(exprU_n, exprS_n)
                D_count_matrix_U = np.concatenate(count_matrix_U_D, axis = 1)
                D_count_matrix_S = np.concatenate(count_matrix_S_D, axis = 1)
                D_count_matrix_total = D_count_matrix_U + D_count_matrix_S

            if outlier_prob!=0 or lib_size_effect_scale!=0 or dropout_percentile!=0:
                print('noised_data')
                adata = ad.AnnData(D_count_matrix_total.T)
                adata.layers['spliced']=D_count_matrix_S.T
                adata.layers['unspliced']=D_count_matrix_U.T
            else:
                print('clean_data')
                adata = ad.AnnData(clean_count_matrix_total.T)
                adata.layers['spliced']=clean_count_matrix_S.T
                adata.layers['unspliced']=clean_count_matrix_U.T



            celllist = []
            for i in range(number_bins):
                lst = [str(i) for _ in range(number_sc)]
                celllist.extend(lst)
            adata.obs['celltype'] = celllist
            adata.var_names = [f'gene{item}' for item in adata.var_names]

            adata.uns['SERGIO_beta'] = pd.DataFrame(SERGIO_beta.T, index= adata.var_names, columns = [str(i) for i in range(number_bins)])
            adata.uns['SERGIO_gamma'] = pd.DataFrame(SERGIO_gamma.T, index= adata.var_names, columns = [str(i) for i in range(number_bins)])
            adata.uns['SERGIO_splice_ratio'] = pd.DataFrame(SERGIO_splice_ratio.T, index= adata.var_names, columns = [str(i) for i in range(number_bins)])


            adata.uns['default_decays']=decays
            adata.uns['default_gamma']=gamma
            adata.uns['default_splice_ratio']=splice_ratio
            
            n_obs=int(adata.n_obs)
            n_vars=int(adata.n_vars)
            all_n=n_obs*n_vars
            s_df=adata.to_df(layer='spliced')
            u_df=adata.to_df(layer='unspliced')
            s_u_df=s_df+u_df
            df_bool_s_u = (s_u_df == 0)
            adata.uns['dropout_rates'] = df_bool_s_u.sum().sum()/all_n
            
            adata_save_path = adata_save_dir + f'/SERGIO_simulation_celltype_bin{number_bins}_sc{number_sc}_spN{int(noise_params*100)}_O{int(outlier_prob*100)}_L{int(lib_size_effect_scale*100)}_D{dropout_percentile}_No{n}.h5ad'
            adata.write(adata_save_path)