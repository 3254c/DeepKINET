import scvelo as scv
import sys
sys.path.append('src/DeepKINET')
import workflow
import utils
import scanpy as sc
from matplotlib import pyplot as plt

adata = scv.datasets.pancreas()
scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
scv.pp.moments(adata)
raw_adata = scv.datasets.pancreas()
adata.layers['spliced'] = raw_adata[:, adata.var_names].layers['spliced']
adata.layers['unspliced'] = raw_adata[:, adata.var_names].layers['unspliced']
color = 'clusters'
adata, vicdyf_exp = workflow.estimate_kinetics(adata, color = color)

print(adata)

utils.embedding_func(adata, color, save_path = '.deepkinet_velocity.png', embeddings = 'X_umap', n_neighbors = 30)

gene_a = 'Actn4'
gene_b = 'Cpe'

fig, axes = plt.subplots(2, 3, figsize=(5 * 3, 5 * 2))
sc.pl.umap(adata, color=gene_a, layer = 'Ms', show=False, ax=axes[0,0], colorbar_loc = None, frameon = False)
axes[0,0].set_title(f'{gene_a} exspression')
sc.pl.umap(adata, color=gene_a, layer = 'each_beta', show=False, ax=axes[0,1], color_map = 'Blues', colorbar_loc = None, frameon = False)
axes[0,1].set_title(f'{gene_a} splicing rates')
sc.pl.umap(adata, color=gene_a, layer = 'each_gamma', show=False, ax=axes[0,2], color_map = 'Reds', colorbar_loc = None, frameon = False)
axes[0,2].set_title(f'{gene_a} degradation rates')
sc.pl.umap(adata, color=gene_b, layer = 'Ms', show=False, ax=axes[1,0], colorbar_loc = None, frameon = False)
axes[1,0].set_title(f'{gene_b} exspression')
sc.pl.umap(adata, color=gene_b, layer = 'each_beta', show=False, ax=axes[1,1], color_map = 'Blues', colorbar_loc = None, frameon = False)
axes[1,1].set_title(f'{gene_b} splicing rates')
sc.pl.umap(adata, color=gene_b, layer = 'each_gamma', show=False, ax=axes[1,2], color_map = 'Reds', colorbar_loc = None, frameon = False)
axes[1,2].set_title(f'{gene_b} degradation rates')

plt.savefig(f"pancreas_{gene_a}_{gene_b}.png")
plt.show()

