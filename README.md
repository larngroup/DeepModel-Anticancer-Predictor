# DeepModel-Anticancer-Predictor
Deep Modelling for Anticancer Drug Response through Genomic Profiles and Compound Structures

We propose a deep neural network model to predict the effect of anticancer drugs in tumors through the half-maximal inhibitory concentration (IC50). The model can be seen as two-fold: first, we pre-trained two autoencoders with high-dimensional gene expression and mutation data to capture the crucial features from tumors; then, this genetic background is translated to cancer cell lines to predict the impact of the genetic variants on a given drug. Moreover, SMILES structures were introduced so that the model can apprehend relevant features regarding the drug compound. Finally, we use drug sensitivity data correlated to the genomic and drugs data to identify features that predict the IC50 value for each pair of drug-cell line. The obtained results demonstrate the effectiveness of the extracted deep representations in the prediction of drug-target interactions

## Requirements:

- Python 3.10.0
- Tensorflow 2.x
- Keras 2.x
- Numpy
- Pandas
- Scikit-learn
- Json
- Pickle
- OS
