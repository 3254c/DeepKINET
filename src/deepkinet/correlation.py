import pandas as pd
import numpy as np

def safe_toarray(x):
    if type(x) != np.ndarray:
        x = x.toarray()
        if not np.all(x == np.floor(x)):
            raise ValueError('target layer of adata should be raw count')
        return x
    else:
        if not np.all(x == np.floor(x)):
            raise ValueError('target layer of adata should be raw count')
        return x

def corr_to_answer(adata, each_beta_layer, each_gamma_layer, mode, cluster_name, answer='dynamo', results = {}):
    autocorr_abs_mean = autocorr(adata, each_beta_layer, each_gamma_layer)
    cell_name = list(adata.obs_names)
    dropout_rate_result = dropout_rates(adata)
    dynamo_beta = adata.uns[f'{answer}_beta']
    dynamo_gamma = adata.uns[f'{answer}_gamma']    
    if each_beta_layer in adata.layers.keys():
        each_beta_df = pd.DataFrame(adata.layers[each_beta_layer], index = adata.obs_names, columns = adata.var_names)
        each_gamma_df= pd.DataFrame(adata.layers[each_gamma_layer], index = adata.obs_names, columns = adata.var_names)
    elif each_beta_layer in adata.uns.keys():
        each_beta_df = adata.uns[each_beta_layer]
        each_gamma_df= adata.uns[each_gamma_layer]
        dynamo_beta = dynamo_beta.T[list(each_beta_df.columns)].T
        dynamo_gamma = dynamo_gamma.T[list(each_beta_df.columns)].T

    s_u_sum_df = pd.DataFrame(safe_toarray(adata.layers['spliced'] + adata.layers['unspliced']),index= adata.obs_names,columns=adata.var_names)
    s_u_sum_df_corr_genes_used_cells_ = s_u_sum_df.sum()[list(dynamo_gamma.index)]
    s_u_sum_df_corr_genes_ = s_u_sum_df_corr_genes_used_cells_.sort_values(ascending=False)

    genes_top0_20_ = list(s_u_sum_df_corr_genes_[0:int(len(s_u_sum_df_corr_genes_)/5)].index)
    genes_top20_40_ = list(s_u_sum_df_corr_genes_[int(len(s_u_sum_df_corr_genes_)/5):int(len(s_u_sum_df_corr_genes_)*2 / 5)].index)
    genes_top40_60_ = list(s_u_sum_df_corr_genes_[int(len(s_u_sum_df_corr_genes_)*2/5):int(len(s_u_sum_df_corr_genes_)*3/ 5)].index)
    genes_top60_80_ = list(s_u_sum_df_corr_genes_[int(len(s_u_sum_df_corr_genes_)*3/5):int(len(s_u_sum_df_corr_genes_)*4/ 5)].index)
    genes_top80_100_ = list(s_u_sum_df_corr_genes_[int(len(s_u_sum_df_corr_genes_)*4/5):int(len(s_u_sum_df_corr_genes_)*5/ 5)].index)

    estimated_beta = pd.DataFrame([], index=dynamo_beta.index, columns=dynamo_beta.columns)
    estimated_gamma = pd.DataFrame([], index=dynamo_beta.index, columns=dynamo_beta.columns)
    for item in dynamo_beta.columns:
        time_type_idx = list(adata.obs[adata.obs[cluster_name]==item].index)
        cells_in_cluster = list(set(time_type_idx)&set(cell_name))
        estimated_beta[item] = each_beta_df[dynamo_beta.index].T[cells_in_cluster].T.mean()
        estimated_gamma[item] = each_gamma_df[dynamo_beta.index].T[cells_in_cluster].T.mean()

    dyn_beta_shared = dynamo_beta[list(estimated_beta.columns)].T[dynamo_beta.index].T
    dyn_gamma_shared = dynamo_gamma[list(estimated_gamma.columns)].T[dynamo_beta.index].T

    adata.var[f'{mode}_{each_beta_layer}_corr'] = dyn_beta_shared.T.corrwith(estimated_beta.T)
    adata.var[f'{mode}_{each_gamma_layer}_corr'] = dyn_gamma_shared.T.corrwith(estimated_gamma.T)

    dyn_beta_corr = dyn_beta_shared.T.corrwith(estimated_beta.T).mean()
    dyn_gamma_corr = dyn_gamma_shared.T.corrwith(estimated_gamma.T).mean()

    dyn_beta_corr_0_20 = dyn_beta_shared.T.corrwith(estimated_beta.T)[genes_top0_20_].mean()
    dyn_beta_corr_20_40 = dyn_beta_shared.T.corrwith(estimated_beta.T)[genes_top20_40_].mean()
    dyn_beta_corr_40_60 = dyn_beta_shared.T.corrwith(estimated_beta.T)[genes_top40_60_].mean()
    dyn_beta_corr_60_80 = dyn_beta_shared.T.corrwith(estimated_beta.T)[genes_top60_80_].mean()
    dyn_beta_corr_80_100 = dyn_beta_shared.T.corrwith(estimated_beta.T)[genes_top80_100_].mean()

    dyn_gamma_corr_0_20 = dyn_gamma_shared.T.corrwith(estimated_gamma.T)[genes_top0_20_].mean()
    dyn_gamma_corr_20_40 = dyn_gamma_shared.T.corrwith(estimated_gamma.T)[genes_top20_40_].mean()
    dyn_gamma_corr_40_60 = dyn_gamma_shared.T.corrwith(estimated_gamma.T)[genes_top40_60_].mean()
    dyn_gamma_corr_60_80 = dyn_gamma_shared.T.corrwith(estimated_gamma.T)[genes_top60_80_].mean()
    dyn_gamma_corr_80_100 = dyn_gamma_shared.T.corrwith(estimated_gamma.T)[genes_top80_100_].mean()


    results_ = {
        'mode':mode,
        'n_genes':len(list(dynamo_beta.index)),
        'beta_corr_mean_all':dyn_beta_corr,
        'beta_corr_mean_top0_20%':dyn_beta_corr_0_20,
        'beta_corr_mean_top20_40%':dyn_beta_corr_20_40,
        'beta_corr_mean_top40_60%':dyn_beta_corr_40_60,
        'beta_corr_mean_top60_80%':dyn_beta_corr_60_80,
        'beta_corr_mean_top80_100%':dyn_beta_corr_80_100,
        'gamma_corr_mean_all':dyn_gamma_corr,
        'gamma_corr_mean_top0_20%':dyn_gamma_corr_0_20,
        'gamma_corr_mean_top20_40%':dyn_gamma_corr_20_40,
        'gamma_corr_mean_top40_60%':dyn_gamma_corr_40_60,
        'gamma_corr_mean_top60_80%':dyn_gamma_corr_60_80,
        'gamma_corr_mean_top80_100%':dyn_gamma_corr_80_100,
        'dropout_rate':dropout_rate_result,
        'autocorr_abs_mean':autocorr_abs_mean
        }
    results.update(results_)
    return results_



def autocorr(adata, layer_1, layer_2):
    if layer_1 in adata.layers.keys():
        df_1 = pd.DataFrame(adata.layers[layer_1], index=adata.obs.index, columns=adata.var.index)
        df_2 = pd.DataFrame(adata.layers[layer_2], index=adata.obs.index, columns=adata.var.index)
    elif layer_1 in adata.uns.keys():
        df_1 = adata.uns[layer_1]
        df_2= adata.uns[layer_2]
    autocorr_abs_mean = df_1.corrwith(df_2).abs().mean()
    print(df_1.corrwith(df_2))
    print('mean', df_1.corrwith(df_2).mean())
    print('abs mean',autocorr_abs_mean)
    return autocorr_abs_mean

def dropout_rates(adata):
    n_obs = int(adata.n_obs)
    n_vars = int(adata.n_vars)
    all_n = n_obs * n_vars
    s_df = adata.to_df(layer='spliced')
    u_df = adata.to_df(layer='unspliced')
    s_u_df = s_df + u_df
    df_bool_s_u = (s_u_df == 0)
    dropout_rate = df_bool_s_u.sum().sum() / all_n
    print('dropout_rate',dropout_rate)
    return dropout_rate