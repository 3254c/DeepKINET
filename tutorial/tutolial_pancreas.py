import sys
import os
sys.path.append('src/deepkinet')
import workflow
import utils
import scanpy as sc
import scvelo as scv

adata = scv.datasets.pancreas()
scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
raw_adata = scv.datasets.pancreas()
adata.layers['spliced'] = raw_adata[:, adata.var_names].layers['spliced']
adata.layers['unspliced'] = raw_adata[:, adata.var_names].layers['unspliced']
adata, deepkinet_exp = workflow.estimate_kinetics(adata)
color = 'clusters'
embeddings = 'X_umap'
utils.embedding_func(adata, color, embeddings = embeddings)
gene_a = 'Actn4'
utils.visualization_kinetics(adata, gene_a, save_path = 'test_a.jpg')
gene_b = 'Cpe'
utils.visualization_kinetics(adata, gene_b, save_path = 'test_b.jpg')