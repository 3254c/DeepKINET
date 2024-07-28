import pandas as pd
import numpy as np
import scvelo as scv
import seaborn as sns
from matplotlib import pyplot as plt
import torch
import exp
import scanpy as sc
import umap
import anndata as ad


def input_checks(adata):
    if (not 'spliced' in adata.layers.keys()) or (not 'unspliced' in adata.layers.keys()):
        raise ValueError(
            f'Input anndata object need to have layers named `spliced` and `unspliced`.')
    if np.sum((adata.layers['spliced'] - adata.layers['spliced'].astype(int)))**2 != 0:
        raise ValueError('layers `spliced` includes non integer number, while count data is required for `spliced`.')

    if np.sum((adata.layers['unspliced'] - adata.layers['unspliced'].astype(int)))**2 != 0:
        raise ValueError('layers `unspliced` includes non integer number, while count data is required for `unspliced`.')


def define_exp(adata, model_params, lr, weight_decay, val_ratio, test_ratio,batch_size, num_workers,checkpoint):
    input_checks(adata)

    if type(adata.layers['spliced']) == np.ndarray:
        s = torch.tensor(adata.layers['spliced'])
    else:
        s = torch.tensor(adata.layers['spliced'].toarray())
    if type(adata.layers['unspliced']) == np.ndarray:
        u = torch.tensor(adata.layers['unspliced'])
    else:
        u = torch.tensor(adata.layers['unspliced'].toarray())

    s = s.float()
    u = u.float()

    deepkinet_exp = exp.DeepKINETExperiment(model_params, lr, weight_decay, s, u, test_ratio, batch_size, num_workers, checkpoint, val_ratio)
    return(deepkinet_exp)

def post_process(adata, deepkinet_exp):
    s = deepkinet_exp.edm.s
    u = deepkinet_exp.edm.u
    batch_onehot = deepkinet_exp.edm.batch_onehot
    
    train_idx = deepkinet_exp.edm.train_idx
    val_idx = deepkinet_exp.edm.validation_idx
    test_idx = deepkinet_exp.edm.test_idx
    adata.uns['train_idx']=train_idx.tolist()
    adata.uns['val_idx']=val_idx.tolist()
    adata.uns['test_idx']=test_idx.tolist()

    deepkinet_exp.device = torch.device('cpu')
    deepkinet_exp.model = deepkinet_exp.model.to(deepkinet_exp.device)

    z, d, qz, qd, s_hat, diff_px_zd_ld, pu_zd_ld  = deepkinet_exp.model(s, u, batch_onehot)
    zl = qz.loc
    d, qd = deepkinet_exp.model.enc_d(zl)
    dl = qd.loc

    norm_mat=deepkinet_exp.edm.norm_mat
    norm_mat_np = norm_mat.cpu().detach().numpy()
    norm_mat_u=deepkinet_exp.edm.norm_mat_u
    norm_mat_u_np = norm_mat_u.cpu().detach().numpy()

    adata.layers['norm_mat'] = norm_mat_np
    adata.layers['norm_mat_u'] = norm_mat_u_np

    if deepkinet_exp.model.batch_key is None:
        px_z_ld = deepkinet_exp.model.dec_z(zl)
        each_beta = deepkinet_exp.model.dec_b(zl) * deepkinet_exp.model.dt
        each_gamma = deepkinet_exp.model.dec_g(zl) * deepkinet_exp.model.dt
        diff_px_zd_ld = deepkinet_exp.model.calculate_diff_x_grad(zl, dl)
        adata.layers['DeepKINET_velocity'] = deepkinet_exp.model.calculate_diff_x_grad(zl, dl).cpu().detach().numpy()
    else:
        px_z_ld = deepkinet_exp.model.dec_z(zl, batch_onehot)
        each_beta = deepkinet_exp.model.dec_b(zl, batch_onehot) * deepkinet_exp.model.dt
        each_gamma = deepkinet_exp.model.dec_g(zl, batch_onehot) * deepkinet_exp.model.dt
        diff_px_zd_ld = deepkinet_exp.model.calculate_diff_x_grad_onehot(zl, batch_onehot, dl)
        adata.layers['DeepKINET_velocity'] = deepkinet_exp.model.calculate_diff_x_grad_onehot(zl, batch_onehot, dl).cpu().detach().numpy()
        
    adata.obs['DeepKINET_velocity'] = np.mean(np.abs(adata.layers['DeepKINET_velocity']), axis=1)

    adata.layers['splicing_rate'] = each_beta.cpu().detach().numpy()
    adata.layers['degradation_rate'] = each_gamma.cpu().detach().numpy()

    adata.obsm['latent_variable'] = zl.cpu().detach().numpy()
    adata.obsm['latent_variable_sample'] = z.cpu().detach().numpy()
    adata.obsm['latent_velocity'] = dl.cpu().detach().numpy()
    adata.obsm['latent_velocity_sample'] = d.cpu().detach().numpy()

    raw_u_ld = (diff_px_zd_ld + s_hat * each_gamma) / each_beta
    pu_zd_ld = raw_u_ld + deepkinet_exp.model.relu(- raw_u_ld).detach()

    adata.layers['pu_zd_ld'] = pu_zd_ld.cpu().detach().numpy()
    px_z_ld_df = pd.DataFrame(px_z_ld.cpu().detach().numpy(),columns=list(adata.var_names))
    pu_zd_ld_df = pd.DataFrame(pu_zd_ld.cpu().detach().numpy(),columns=list(adata.var_names))
    adata.layers['u_hat'] = pu_zd_ld_df
    adata.layers['s_hat'] = px_z_ld.cpu().detach().numpy()
    adata.obsm['s_hat'] = px_z_ld.cpu().detach().numpy()
    s_df = pd.DataFrame(s.cpu().detach().numpy(), columns=list(adata.var_names))
    u_df = pd.DataFrame(u.cpu().detach().numpy(), columns=list(adata.var_names))

    s_correlation=(px_z_ld_df).corrwith(s_df / norm_mat_np).mean()
    u_correlation = (pu_zd_ld_df).corrwith(u_df / norm_mat_u_np).mean()

    train_s_correlation = ((px_z_ld_df).T[train_idx].T).corrwith((s_df / norm_mat_np).T[train_idx].T).mean()
    train_u_correlation = ((pu_zd_ld_df).T[train_idx].T).corrwith((u_df / norm_mat_u_np).T[train_idx].T).mean()
    val_s_correlation = ((px_z_ld_df).T[val_idx].T).corrwith((s_df / norm_mat_np).T[val_idx].T).mean()
    val_u_correlation = ((pu_zd_ld_df).T[val_idx].T).corrwith((u_df / norm_mat_u_np).T[val_idx].T).mean()
    test_s_correlation = ((px_z_ld_df).T[test_idx].T).corrwith((s_df / norm_mat_np).T[test_idx].T).mean()
    test_u_correlation = ((pu_zd_ld_df).T[test_idx].T).corrwith((u_df / norm_mat_u_np).T[test_idx].T).mean()

    print('train_s_correlation', train_s_correlation)
    print('train_u_correlation', train_u_correlation)

    print('val_s_correlation', val_s_correlation)
    print('val_u_correlation', val_u_correlation)

    print('test_s_correlation', test_s_correlation)
    print('test_u_correlation', test_u_correlation)

    results_dict = {
        'train_s_correlation':train_s_correlation, 'train_u_correlation':train_u_correlation,
        'val_s_correlation':val_s_correlation, 'val_u_correlation':val_u_correlation,
        'test_s_correlation':test_s_correlation, 'test_u_correlation':test_u_correlation,
        'Dynamics_last_val_loss':adata.uns['Dynamics_last_val_loss'].item(),
        'Dynamics_last_test_loss':adata.uns['Dynamics_last_test_loss'].item(),
        'Kinetics_last_val_loss':adata.uns['Kinetics_last_val_loss'].item(),
        'Kinetics_last_test_loss':adata.uns['Kinetics_last_test_loss'].item()}
    return results_dict

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

def embed_z(z_mat, n_neighbors, min_dist):
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist)
    z_embed = reducer.fit_transform(z_mat)
    return(z_embed)

def plt_beta_gamma(adata):
    fig=plt.figure(figsize=(30,30))
    for i,name in enumerate(adata.layers['each_beta'].T[:16]):
        ax=plt.subplot(4,4,i+1)
        ax.set_xlabel('each_beta of {}'.format(adata.var_names[i]))
        ax.set_ylim(0, 5)
        sns.violinplot(y=name,orient='v',ax=ax).set_title('gene={}'.format(adata.var_names[i]))

    fig=plt.figure(figsize=(30,30))
    for i,name in enumerate(adata.layers['each_gamma'].T[:16]):
        ax=plt.subplot(4,4,i+1)
        ax.set_xlabel('each_gamma of {}'.format(adata.var_names[i]))
        ax.set_ylim(0, 5)
        sns.violinplot(y=name,orient='v',ax=ax).set_title('gene={}'.format(adata.var_names[i]))

    fig=plt.figure(figsize=(30,30))
    for i,name in enumerate(adata.layers['lambda'].T[:16]):
        ax=plt.subplot(4,4,i+1)
        ax.set_xlabel('DeepKINET_expression of {}'.format(adata.var_names[i]))
        ax.set_ylim(0, 5)
        sns.violinplot(y=name,orient='v',ax=ax).set_title('gene={}'.format(adata.var_names[i]))

    fig=plt.figure(figsize=(30,30))
    for i,name in enumerate(adata.layers['DeepKINET_velocity'].T[:16]):
        ax=plt.subplot(4,4,i+1)
        ax.set_xlabel('DeepKINET_velocity of {}'.format(adata.var_names[i]))
        ax.set_ylim(-0.03, 0.03)
        sns.violinplot(y=name,orient='v',ax=ax).set_title('gene={}'.format(adata.var_names[i]))


def embedding_func(adata, color, save_path = '.latent_velocity.png', embeddings = 'X_umap', n_neighbors = 30):

    adata_z = ad.AnnData(adata.obsm['latent_variable'])
    adata_z.obs_names = adata.obs_names
    adata_z.layers['latent_variable'] = adata.obsm['latent_variable']
    adata_z.obsm['latent_variable'] = adata.obsm['latent_variable']
    if color != None:
        adata_z.obs[f'{color}'] = adata.obs[f'{color}']
    adata_z.layers['latent_velocity'] = adata.obsm['latent_velocity']
    adata_z.obsm['latent_velocity'] = adata.obsm['latent_velocity']
    adata_z.obsm['X_original_umap'] = adata.obsm[embeddings]
    adata.obsm['X_original_umap'] = adata.obsm[embeddings]

    #dの図示
    sc.pp.neighbors(adata_z, n_neighbors = n_neighbors, use_rep='latent_variable')
    scv.tl.velocity_graph(adata_z,vkey='latent_velocity',xkey='latent_variable')
    scv.pl.velocity_embedding_grid(adata_z, basis='X_original_umap',X=adata_z.obsm['X_original_umap'],vkey='latent_velocity', width=0.002, arrow_length=1,headwidth=10, density=0.4, arrow_color='black', color=color, save = save_path)

def kinetic_rate_cluster_separate(adata):
  #cluster解析
  sc.pp.neighbors(adata, n_neighbors=30, use_rep='latent_variable')

  splicing_rate_z_np = (adata.layers['splicing_rate'] - adata.layers['splicing_rate'].mean(axis = 0)) / adata.layers['splicing_rate'].std(axis = 0)
  degradation_rate_z_np = (adata.layers['degradation_rate'] - adata.layers['degradation_rate'].mean(axis = 0)) /adata.layers['degradation_rate'].std(axis = 0)

  adata_splicing_rate_T = ad.AnnData(splicing_rate_z_np.T)
  adata_splicing_rate_T.obs_names = adata.var_names
  sc.pp.pca(adata_splicing_rate_T)
  sc.pp.neighbors(adata_splicing_rate_T)
  sc.tl.leiden(adata_splicing_rate_T, key_added = 'splicing_rate_leiden')
  #sc.tl.umap(adata_splicing_rate_T)
  #sc.pl.umap(adata_splicing_rate_T, color = 'splicing_rate_leiden')
  adata.var['splicing_rate_leiden'] = adata_splicing_rate_T.obs['splicing_rate_leiden']

  adata_degradation_rate_T = ad.AnnData(degradation_rate_z_np.T)
  adata_degradation_rate_T.obs_names = adata.var_names
  sc.pp.pca(adata_degradation_rate_T)
  sc.pp.neighbors(adata_degradation_rate_T)
  sc.tl.leiden(adata_degradation_rate_T, key_added = 'degradation_rate_leiden')
  #sc.tl.umap(adata_degradation_rate_T)
  #sc.pl.umap(adata_degradation_rate_T, color = 'degradation_rate_leiden')
  adata.var['degradation_rate_leiden'] = adata_degradation_rate_T.obs['degradation_rate_leiden']

def kinetic_rate_cluster_both(adata):
  #cluster解析
  sc.pp.neighbors(adata, n_neighbors=30, use_rep='latent_variable')

  splicing_rate_z_np = (adata.layers['splicing_rate'] - adata.layers['splicing_rate'].mean(axis = 0)) / adata.layers['splicing_rate'].std(axis = 0)
  adata.layers['splicing_rate_z'] = splicing_rate_z_np
  degradation_rate_z_np = (adata.layers['degradation_rate'] - adata.layers['degradation_rate'].mean(axis = 0)) /adata.layers['degradation_rate'].std(axis = 0)
  adata.layers['degradation_rate_z'] = degradation_rate_z_np

  kinetic_rate_np_T = np.concatenate([splicing_rate_z_np.T, degradation_rate_z_np.T], axis =1)
  adata_kinetic_rate_T = ad.AnnData(kinetic_rate_np_T)
  adata_kinetic_rate_T.obs_names = adata.var_names
  sc.pp.pca(adata_kinetic_rate_T)
  sc.pp.neighbors(adata_kinetic_rate_T)
  sc.tl.leiden(adata_kinetic_rate_T, key_added = 'kinetic_rate_leiden')
  #sc.tl.umap(adata_kinetic_rate_T)
  #sc.pl.umap(adata_kinetic_rate_T, color = 'kinetic_rate_leiden')
  adata.var['kinetic_rate_leiden'] = adata_kinetic_rate_T.obs['kinetic_rate_leiden']

def rank_genes_groups_splicing(adata, groupby, groups, reference, method = 't-test', n_genes=20):
  print('Ranking genes by splicing rates')
  sc.tl.rank_genes_groups(adata, groupby = groupby, layer = 'splicing_rate', groups=groups, reference=reference, method=method, key_added = 'rank_genes_groups_splicing')
  sc.pl.rank_genes_groups(adata, groups = groups, n_genes=n_genes, show = False, key = 'rank_genes_groups_splicing')
  rank_genes_splicing = list(adata.uns['rank_genes_groups_splicing']['names'][groups[0]])
  return rank_genes_splicing

def rank_genes_groups_degradation(adata, groupby, groups, reference, method = 't-test', n_genes=20):
  print('Ranking genes by degradation rates')
  sc.tl.rank_genes_groups(adata, groupby = groupby, layer = 'degradation_rate', groups=groups, reference=reference, method=method, key_added = 'rank_genes_groups_degradation')
  sc.pl.rank_genes_groups(adata, groups = groups, n_genes=n_genes, show = False, key = 'rank_genes_groups_degradation')
  rank_genes_degradation = list(adata.uns['rank_genes_groups_degradation']['names'][groups[0]])
  return rank_genes_degradation

def visualization_kinetics(adata, gene, save_path=None):
  fig, axes = plt.subplots(1, 3, figsize=(5 * 3, 5 * 1))
  sc.pl.umap(adata, color=gene, layer = 's_hat', show=False, ax=axes[0], colorbar_loc = None, frameon = False)
  axes[0].set_title(f'{gene} exspression')
  sc.pl.umap(adata, color=gene, layer = 'splicing_rate', show=False, ax=axes[1], color_map = 'Blues', colorbar_loc = None, frameon = False)
  axes[1].set_title(f'{gene} splicing rates')
  sc.pl.umap(adata, color=gene, layer = 'degradation_rate', show=False, ax=axes[2], color_map = 'Reds', colorbar_loc = None, frameon = False)
  axes[2].set_title(f'{gene} degradation rates')
  if save_path != None:
    fig.savefig(save_path)