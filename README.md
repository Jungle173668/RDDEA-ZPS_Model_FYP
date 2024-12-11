# Data-driven Evolution Algorithm for Solving Non-linear Systems of Equations
Radial basis function based Data-Driven Evolution Algorithm & Zero Point Sampling (RDDEA-ZPS) Model. This is the final year project in Harbin Institute of Technology, Shenzhen. Author: Zhao C J. Supervisor: Zhang X M.

This is source code of **RDDEA-ZPS** Model.

In this model, I investigated solving nonlinear equations using Data-Driven  Evolutionary Algorithms (DDEA), especially the WDDEA-DBC model. Enhanced the model and developed the RDDEA-ZPS algorithm.

The **main contribution** are as follows:  
(1) Built the baseline model WDDEA-DBC by PyTorch. Modified the structure of the RBFNN model to fit the inverse operator of non-linear systems of equations, reducing the error introduced by the norm.
(2) Improved the data generation method by Latin Hypercube Sampling, Data-Based Clustering and Zero-Point Sampling to create higher-quality datasets.
(3) Shifted the application scenario of the FPA algorithm, integrated the surrogate models by MSE-weighted sampling.

The RDDEA-ZPS model reduced the Euler distance error by an average of 80.89% and improved stability by 63.77%.
