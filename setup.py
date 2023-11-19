import setuptools

project_urls = {
  'DeepKINET': 'https://github.com/3254c/DeepKINET'
}

setuptools.setup(
    name="DeepKINET",
    version="0.0.1",
    author="Chikara Mizukoshi",
    author_email="mizucoscos@gmail.com",
    description="Estimate mRNA kinetics at the single-cell level",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7",
    install_requires = ['anndata==0.10.3',
                        'scanpy==1.9.6',
                        'einops==0.7.0',
                        'leidenalg==0.10.1',
                        'scvelo==0.2.5',
                        'umap-0.1.1',
                        'umap-learn==0.5.5',
                        'torch==2.1.0+cu118',
                        'pandas==1.5.3',
                        'numpy==1.23.5',
                        'seaborn==0.12.2',
                        'matplotlib==3.7.1'
                        ]
)
