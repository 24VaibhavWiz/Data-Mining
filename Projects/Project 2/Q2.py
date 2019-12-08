from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.utils.metric import distance_metric, type_metric
import matplotlib.pyplot as plot
import numpy as np

data_set_raw = [(2, 10), (4, 7), (3, 12), (5, 11), (2, 5), (6, 13), (4,
7), (7, 14), (8, 12), (3, 10),
(9, 6), (5, 7), (4, 13), (6, 16), (8, 15)]

	centers = kmeans_plusplus_initializer(data_set_raw, 3).initialize()
	eudi = distance_metric(type_metric.EUCLIDEAN)
	# Saved Figure_1

	km = kmeans(data_set_raw, centers, metric=eudi)
	km.process()
	f_center = km.get_centers()
	f_cluster = km.get_clusters()
	
	print(f_cluster, f_center)
	ax = plot.subplot(111)
	array = np.array(data_set_raw)
	f_center = np.array(f_center)
	
	plot.title('K-means')
	plot.scatter(array[f_cluster[0][:]][:, 0],
	array[f_cluster[0][:]][:, 1],
	s=50,
	c='lightgreen',
	marker='s',
	
	label='Category 1')
	plot.scatter(array[f_cluster[1][:]][:, 0],
	array[f_cluster[1][:]][:, 1],
	s=50,
	c='orange',
	marker='o',

	label='Category 2')
	plot.scatter(array[f_cluster[2][:]][:, 0],
	array[f_cluster[2][:]][:, 1],
	s=50,
	c='yellow',
	marker='v',

	label='Category 3')
	plot.scatter(f_center[:][:, 0],
	f_center[:][:, 1],s=60,
	c='red',
	marker='*',
	
	label='Centers')
	plot.legend(loc='center left',
	scatterpoints=1, bbox_to_anchor=(1, 0.5))
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

	plot.show()
	plot.show()
