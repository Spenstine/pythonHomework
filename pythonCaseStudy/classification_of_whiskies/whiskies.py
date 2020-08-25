import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster.bicluster import SpectralCoclustering


'''importing data'''
#from prior observation I know the first row is header so I let the default header setting
#I also get to know the indices are the RowID column which is at index 0
#whisky = pd.read_csv('whiskies.txt', index_col = 'RowID') #same to index_col = 0

#ignore the index_col because of the rearrangements we'll do as we model our spectral co-cluster
whisky = pd.read_csv('whiskies.txt')

whisky['Region'] = pd.read_csv('regions.txt')

#flavors are categorized into all the columns from 'Body' to 'Floral'
flavor = whisky.loc[:, 'Body':'Floral'] #iloc also works but for integer indices and slices

#we get the pearson correlation using pd.DataFrame.corr() function
#result is a correlation matrix
corr_flavor = pd.DataFrame.corr(flavor)


'''correlations between the data'''
#visualization of the correlation between flavors
plt.figure(figsize = (10, 10))
plt.pcolor(corr_flavor)
plt.colorbar()
plt.savefig('corr_flavors.png')

corr_whisky = pd.DataFrame.corr(flavor.transpose())
plt.figure(figsize = (10, 10))
plt.pcolor(corr_whisky)
plt.colorbar()
plt.savefig('corr_whiskies.png')


'''spectral co-clustering'''
#spectral co-clustering deals with eigenvectors and eigenvalues
#we use sklearn to create a model that will be useful to us 
model = SpectralCoclustering(n_clusters=6, random_state=0)
model.fit(corr_whisky)

#veiw the row clusters
#expectation: boolean list of lists, six items for clusters(6) in outer list, inner lists are observations(86) in the clusters
model.rows_

#count the observations in each cluster by summing the columns in our model
#expectation: list of six items. each item represents the number of observations in the cluster, index corresponds to cluster
np.sum(model.rows_, axis = 1)

#count the clusters for each observation by summing the rows in our model
#expectation: list of ones. items in the list represent the observations(86) and each observation has a cluster
np.sum(model.rows_, axis = 0)


'''comparing correlation matrices'''
whisky['Group'] = pd.Series(model.row_labels_, index = whisky.index)
whisky = whisky.iloc[np.argsort(model.row_labels_)]
whisky = whisky.reset_index(drop = True)

correlations = pd.DataFrame.corr(whisky.iloc[:, 2:14].transpose())
correlations = np.array(correlations)

plt.figure(figsize = (14, 7))
plt.subplot(121)
plt.title('Original')
plt.pcolor(corr_whisky)
plt.subplot(122)
plt.title('Clustered')
plt.pcolor(correlations)
plt.colorbar()
plt.savefig('compare_corelations.png')