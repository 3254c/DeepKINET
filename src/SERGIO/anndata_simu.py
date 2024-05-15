import scanpy as sc

adata = sc.read_h5ad('/home/mizukoshi/deepKINET_develop/SERGIO/data_bin6_D65/SERGIO_simulation_celltype_bin6_sc25_spN0_O0_L0_D65_No0.h5ad')


print(adata.var_names)
print(adata.uns['SERGIO_beta'])
print(adata.uns['SERGIO_gamma'])
print(adata.uns['SERGIO_splice_ratio'])

# adata = sc.read_h5ad('/home/mizukoshi/DeepKINET/src/SERGIO/data/_DS7_simulation_celltype_spN0_O0_L0_D0_No0.h5ad')
# print(adata)
# print(adata.uns['SERGIO_beta'])
# print(adata.uns['SERGIO_gamma'])
# print(adata.uns['SERGIO_splice_ratio'])
