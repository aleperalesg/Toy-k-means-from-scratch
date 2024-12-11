import numpy as np 
import matplotlib.pyplot as plt 


## get distances between centroid and points
def distance(data,seeds):
	return np.sqrt(np.sum((seeds[:, np.newaxis, :] - data[np.newaxis, :, :])**2, axis=2))

## get centroid
def centroid(points):
	return np.mean(points, axis=0)


## create fake data groups
mean  = [0, 0]
cov   = [[1, 0], [0, 1]]
mean2 = [3,4]
cov2  = [[1, 0], [0, 1]]
mean3 = [-4,4]
cov3  = [[1, 0], [0, 1]]

x, y = np.random.multivariate_normal(mean, cov, 40).T
x2, y2 = np.random.multivariate_normal(mean2, cov2, 40).T
x3, y3 = np.random.multivariate_normal(mean3, cov3, 40).T

X = np.concatenate((x,x2,x3))
Y = np.concatenate((y,y2,y3))
data = np.stack((X,Y),axis = 1)


## set number of clusters and iterations
num_c = 3
itera = 30 

## set and create centorid seeds
centroids = np.zeros((num_c,2))

for i in range(num_c):
	centroids[i,:] = np.random.uniform(low = -6, high = 8, size = 2)


## get clusters
for i in range(itera):

	## get distances and label each point with the closest centroid
	dist = distance(data, centroids)
	logical =  np.argmax((dist == np.min(dist, axis=0, keepdims=True)).astype(int),axis = 0)


	clstr = []
	for idx,i in enumerate(np.unique(logical)):

		## get index of centroid 
		ind = [index for index, value in enumerate(logical) if value == i]

		## get points
		new_data = data[ind]

		## update centroids
		new_centroid = centroid(new_data)
		centroids[idx,:] = new_centroid

		## save cluster
		clstr.append(new_data)



## plot data and clusters
colors = ['red','blue','green']
fig, ax = plt.subplots(1, 2)

for idx, i in enumerate(clstr):

	ax[1].scatter(i[:,0],i[:,1],color = colors[idx])
	ax[1].scatter(centroids[idx,0],centroids[idx,1],marker='+',color = colors[idx])

ax[0].scatter(X,Y,color = 'gray')
plt.show()












