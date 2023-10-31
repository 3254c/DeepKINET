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


def define_exp(adata, model_params, lr, val_ratio, test_ratio,batch_size, num_workers,checkpoint):
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

    vicdyf_exp = exp.DeepKINETExperiment(model_params, lr, s, u, test_ratio, batch_size, num_workers, checkpoint, val_ratio)
    return(vicdyf_exp)

def post_process(adata, vicdyf_exp):
    s = vicdyf_exp.edm.s
    u = vicdyf_exp.edm.u
    val_idx = vicdyf_exp.edm.validation_idx
    test_idx = vicdyf_exp.edm.test_idx
    adata.uns['val_idx']=val_idx.tolist()
    adata.uns['test_idx']=test_idx.tolist()
    vicdyf_exp.device = torch.device('cpu')
    vicdyf_exp.model = vicdyf_exp.model.to(vicdyf_exp.device)
    z, dz, qz, qdz, s_hat, diff_px_zd_ld, pu_zd_ld  = vicdyf_exp.model(s, u)
    zl = qz.loc
    dz, qdz = vicdyf_exp.model.enc_d(zl)
    px_z_ld = vicdyf_exp.model.dec_z(zl)
    adata.layers['lambda'] = px_z_ld.cpu().detach().numpy()
    norm_mat=vicdyf_exp.edm.norm_mat
    norm_mat_np = norm_mat.cpu().detach().numpy()
    norm_mat_u=vicdyf_exp.edm.norm_mat_u
    norm_mat_u_np = norm_mat_u.cpu().detach().numpy()

    adata.layers['norm_mat'] = norm_mat_np
    adata.layers['norm_mat_u'] = norm_mat_u_np
    t_sum_per_cells = (s+u).sum(axis=1)
    adata.obs['total_counts']=t_sum_per_cells

    beta = vicdyf_exp.model.softplus(vicdyf_exp.model.logbeta)* vicdyf_exp.model.dt
    gamma = vicdyf_exp.model.softplus(vicdyf_exp.model.loggamma)* vicdyf_exp.model.dt
    each_beta = vicdyf_exp.model.dec_b(zl) * vicdyf_exp.model.dt
    each_gamma = vicdyf_exp.model.dec_g(zl) * vicdyf_exp.model.dt
    adata.var['beta'] = (beta).cpu().detach().numpy()
    adata.var['gamma'] = (gamma).cpu().detach().numpy()
    adata.layers['each_beta'] = each_beta.cpu().detach().numpy()
    adata.layers['each_gamma'] = each_gamma.cpu().detach().numpy()

    dzl = qdz.loc

    adata.obsm['X_vicdyf_zl'] = zl.cpu().detach().numpy()
    adata.obsm['X_vicdyf_z'] = z.cpu().detach().numpy()
    adata.obsm['X_vicdyf_dl'] = dzl.cpu().detach().numpy()
    adata.obsm['X_vicdyf_d'] = dz.cpu().detach().numpy()

    diff_px_zd_ld = vicdyf_exp.model.calculate_diff_x_grad(zl, dzl)

    raw_u_ld = (diff_px_zd_ld + s_hat * each_gamma) / each_beta
    pu_zd_ld = raw_u_ld + vicdyf_exp.model.relu(- raw_u_ld).detach()

    adata.layers['pu_zd_ld'] = pu_zd_ld.cpu().detach().numpy()
    px_z_ld_df = pd.DataFrame(px_z_ld.cpu().detach().numpy(),columns=list(adata.var_names))
    pu_zd_ld_df = pd.DataFrame(pu_zd_ld.cpu().detach().numpy(),columns=list(adata.var_names))
    adata.layers['u_hat'] = pu_zd_ld_df
    adata.layers['s_hat_normmat'] = (px_z_ld * norm_mat).cpu().detach().numpy()
    adata.layers['s_hat'] = px_z_ld_df
    adata.obsm['s_hat'] = px_z_ld.cpu().detach().numpy()
    adata.obsm['s_hat_normmat'] = (px_z_ld * norm_mat).cpu().detach().numpy()
    s_df = pd.DataFrame(s.cpu().detach().numpy(), columns=list(adata.var_names))
    u_df = pd.DataFrame(u.cpu().detach().numpy(), columns=list(adata.var_names))

    s_correlation=(px_z_ld_df).corrwith(s_df / norm_mat_np).mean() #遺伝子方向になっている。
    u_correlation = (pu_zd_ld_df).corrwith(u_df / norm_mat_u_np).mean()
    val_s_correlation = ((px_z_ld_df).T[val_idx].T).corrwith((s_df / norm_mat_np).T[val_idx].T).mean()
    val_u_correlation = ((pu_zd_ld_df).T[val_idx].T).corrwith((u_df / norm_mat_u_np).T[val_idx].T).mean()
    test_s_correlation = ((px_z_ld_df).T[test_idx].T).corrwith((s_df / norm_mat_np).T[test_idx].T).mean()
    test_u_correlation = ((pu_zd_ld_df).T[test_idx].T).corrwith((u_df / norm_mat_u_np).T[test_idx].T).mean()

    adata.uns['s_correlation']=s_correlation
    adata.uns['u_correlation']=u_correlation
    print('s_correlation', adata.uns['s_correlation'])
    print('u_correlation', adata.uns['u_correlation'])

    adata.uns['val_s_correlation']=val_s_correlation
    adata.uns['val_u_correlation']=val_u_correlation
    print('val_s_correlation', adata.uns['val_s_correlation'])
    print('val_u_correlation', adata.uns['val_u_correlation'])

    adata.uns['test_s_correlation']=test_s_correlation
    adata.uns['test_u_correlation']=test_u_correlation
    print('test_s_correlation', adata.uns['test_s_correlation'])
    print('test_u_correlation', adata.uns['test_u_correlation'])

    adata.layers['vicdyf_mean_velocity'] = vicdyf_exp.model.calculate_diff_x_grad(zl, dzl).cpu().detach().numpy()
    adata.layers['vicdyf_fluctuation'] = vicdyf_exp.model.calculate_diff_x_std(zl, qdz.scale).cpu().detach().numpy()
    adata.obs['vicdyf_fluctuation'] = np.mean(adata.layers['vicdyf_fluctuation'], axis=1)
    adata.obs['vicdyf_mean_velocity'] = np.mean(np.abs(adata.layers['vicdyf_mean_velocity']), axis=1)

    n_obs = int(adata.n_obs)
    n_vars = int(adata.n_vars)
    all_n = n_obs * n_vars
    s_df = adata.to_df(layer='spliced')
    u_df = adata.to_df(layer='unspliced')
    s_u_df = s_df + u_df
    df_bool_s_u = (s_u_df == 0)
    dropout_rate = df_bool_s_u.sum().sum() / all_n
    print('dropout_rate',dropout_rate)

    results_dict = {
        's_correlation':adata.uns['s_correlation'], 'u_correlation':adata.uns['u_correlation'],
        'val_s_correlation':adata.uns['val_s_correlation'], 'val_u_correlation':adata.uns['val_u_correlation'],
        'test_s_correlation':adata.uns['test_s_correlation'], 'test_u_correlation':adata.uns['test_u_correlation'],
        'Dynamics_last_val_loss':adata.uns['Dynamics_last_val_loss'].item(),
        'Dynamics_last_test_loss':adata.uns['Dynamics_last_test_loss'].item(),
        'Kinetics_last_val_loss':adata.uns['Kinetics_last_val_loss'].item(),
        'Kinetics_last_test_loss':adata.uns['Kinetics_last_test_loss'].item(),
        'n_obs':float(n_obs), 'n_vars':float(n_vars),
        'dropout_rate':dropout_rate}
    return results_dict

def embed_z(z_mat, n_neighbors, min_dist):
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist)
    z_embed = reducer.fit_transform(z_mat)
    return(z_embed)

def plot_velocity_kojima(adata, cluster_key, min_dist = 0.1, n_neighbors = 30):
    z_embed = embed_z(adata.obsm['X_vicdyf_zl'], n_neighbors, min_dist)
    adata.obsm['X_kojima_umap'] = z_embed

    adata_z = ad.AnnData(adata.obsm['X_vicdyf_zl'])
    adata_z.obs_names = adata.obs_names
    adata_z.layers['X_vicdyf_zl'] = adata.obsm['X_vicdyf_zl']
    adata_z.obsm['X_vicdyf_zl'] = adata.obsm['X_vicdyf_zl']
    if cluster_key != None:
        adata_z.obs[f'{cluster_key}'] = adata.obs[f'{cluster_key}']
    adata_z.layers['X_vicdyf_dl'] = adata.obsm['X_vicdyf_dl']
    adata_z.obsm['X_vicdyf_dl'] = adata.obsm['X_vicdyf_dl']
    adata_z.obsm['X_kojima_umap'] = adata.obsm['X_kojima_umap']

    #ds/dt
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X_vicdyf_zl')
    scv.tl.velocity_graph(adata,vkey='vicdyf_mean_velocity',xkey='s_hat')
    scv.pl.velocity_embedding_stream(adata, basis='X_kojima_umap',X=adata.obsm['X_kojima_umap'],vkey='vicdyf_mean_velocity', color=cluster_key)
    scv.pl.velocity_embedding_grid(adata, basis='X_kojima_umap',X=adata.obsm['X_kojima_umap'],vkey='vicdyf_mean_velocity', width=0.002, arrow_length=1,headwidth=10, density=0.3, arrow_color='black', color=cluster_key)

    #dの図示
    sc.pp.neighbors(adata_z, n_neighbors=n_neighbors, use_rep='X_vicdyf_zl')
    scv.tl.velocity_graph(adata_z,vkey='X_vicdyf_dl',xkey='X_vicdyf_zl')
    scv.pl.velocity_embedding_stream(adata_z, basis='X_kojima_umap',X=adata_z.obsm['X_kojima_umap'],vkey='X_vicdyf_dl', color=cluster_key)
    scv.pl.velocity_embedding_grid(adata_z, basis='X_kojima_umap',X=adata_z.obsm['X_kojima_umap'],vkey='X_vicdyf_dl', width=0.002, arrow_length=1,headwidth=10, density=0.3, arrow_color='black', color=cluster_key)


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
        ax.set_xlabel('vicdyf_expression of {}'.format(adata.var_names[i]))
        ax.set_ylim(0, 5)
        sns.violinplot(y=name,orient='v',ax=ax).set_title('gene={}'.format(adata.var_names[i]))

    fig=plt.figure(figsize=(30,30))
    for i,name in enumerate(adata.layers['vicdyf_mean_velocity'].T[:16]):
        ax=plt.subplot(4,4,i+1)
        ax.set_xlabel('vicdyf_mean_velocity of {}'.format(adata.var_names[i]))
        ax.set_ylim(-0.03, 0.03)
        sns.violinplot(y=name,orient='v',ax=ax).set_title('gene={}'.format(adata.var_names[i]))


def embedding_func(adata, color, save_path = '.deepkinet_velocity.png', embeddings = 'X_umap', n_neighbors = 30):

    adata_z = ad.AnnData(adata.obsm['X_vicdyf_zl'])
    adata_z.obs_names = adata.obs_names
    adata_z.layers['X_vicdyf_zl'] = adata.obsm['X_vicdyf_zl']
    adata_z.obsm['X_vicdyf_zl'] = adata.obsm['X_vicdyf_zl']
    if color != None:
        adata_z.obs[f'{color}'] = adata.obs[f'{color}']
    adata_z.layers['X_vicdyf_dl'] = adata.obsm['X_vicdyf_dl']
    adata_z.obsm['X_vicdyf_dl'] = adata.obsm['X_vicdyf_dl']
    adata_z.obsm['X_original_umap'] = adata.obsm[embeddings]
    adata.obsm['X_original_umap'] = adata.obsm[embeddings]

    #dの図示
    sc.pp.neighbors(adata_z, n_neighbors = n_neighbors, use_rep='X_vicdyf_zl')
    scv.tl.velocity_graph(adata_z,vkey='X_vicdyf_dl',xkey='X_vicdyf_zl')
    scv.pl.velocity_embedding_grid(adata_z, basis='X_original_umap',X=adata_z.obsm['X_original_umap'],vkey='X_vicdyf_dl', width=0.002, arrow_length=1,headwidth=10, density=0.4, arrow_color='black', color=color, save = save_path)

    # #ds/dt
    # sc.pp.neighbors(adata, n_neighbors = n_neighbors, use_rep='X_vicdyf_zl')
    # scv.tl.velocity_graph(adata,vkey='vicdyf_mean_velocity',xkey='s_hat')
    # scv.pl.velocity_embedding_grid(adata, basis='X_original_umap',X=adata.obsm['X_original_umap'],vkey='vicdyf_mean_velocity', width=0.002, arrow_length=1,headwidth=10, density=0.4, arrow_color='black', color=color)

    #sc.pl.umap(adata, color='total_counts',neighbors_key='X_vicdyf_zl')
    sc.pl.umap(adata,color='vicdyf_fluctuation')
