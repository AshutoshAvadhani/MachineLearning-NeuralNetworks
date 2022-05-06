
import numpy as np
from collections import defaultdict
from numpy.lib.arraysetops import intersect1d
from numpy.lib.function_base import blackman
from numpy.lib.index_tricks import index_exp
from numpy.lib.polynomial import polymul
from numpy.linalg import norm
import pandas as pd

def euclidean_dist(x1,x2):
    dist = np.sqrt(np.sum((x1-x2)**2))
    return dist

def cosine_dist(x1,x2):
    #dist = (x1 * x2)/((np.sqrt(x1^2))+(np.sqrt(x2^2)))
    dist = np.dot(x1,x2)/(norm(x1)*norm(x2))
    return dist

def Jaccard_dist(x1,x2):

    sum_num = 0.00
    sum_deno = 0.00

    for i in range(x1.shape[0]):
        sum_num = np.min(np.array([x1[i],x2[i]]))
        sum_deno = np.max(np.array([x1[i],x2[i]]))
    
    if sum_deno == 0:
        dist = 1
    else:
        dist = 1 - (sum_num/sum_deno)

    intersection_x1_x2 = np.logical_and(x1,x2)
    union_x1_x2 = np.logical_or(x1,x2)
    dist = intersection_x1_x2.sum() / float(union_x1_x2.sum())
    return dist

class KMeans_Class:

    def __init__(self,k=3,iterations_count = 1000):
        self.k = k
        self.iterations_count = iterations_count

        self.create_clusters = [[] for i in range(self.k)]
        self.create_centroid=[]
        self.SSE_Val_List = []

    def Func_create_centroid(self, arr_data):
        
        self.arr_data = arr_data
        self.num_samples = arr_data.shape[0]
        self.num_features = arr_data.shape[1]
        

        random_centroid = np.random.choice(self.num_features,self.k) #replace = false because we do not want to have same sample again
        
        self.create_centroid = [self.arr_data[i] for i in random_centroid]

        for i in range (self.iterations_count):
            self.clusters = self.func_create_clusters(self.create_centroid)
            centroids_old = self.create_centroid
            self.create_centroid = self.func_Getcentroids(self.clusters)

            if self.func_AreConverged(centroids_old,self.create_centroid):
                print("The iterations completed are --> " + str(self.iterations_count))
                break
            
        return self.func_Get_cluster_label(self.clusters)


    def func_Get_cluster_label(self, clusters):
        SSE_values = 0
        labels = np.empty(self.num_samples)

        for key in range(self.k):
            cluster_key = self.clusters[key]
            ce = self.create_centroid[key]
            for i in cluster_key:
                SSE_values += euclidean_dist(ce,i)
                
        self.SSE_Val_List.append(SSE_values)

        for index_cluster, cluster in enumerate(clusters):
            for sample_index in cluster:
                labels[sample_index] = index_cluster
        return labels

    def func_AreConverged(self,old_Centroid,new_Centroid):

        # The distances calculation calling functions are as follows :--
        distances = [euclidean_dist(old_Centroid[i],new_Centroid[i]) for i in range(self.k)]
        #distances = [cosine_dist(old_Centroid[i],new_Centroid[i]) for i in range(self.k)]
        #distances = [Jaccard_dist(old_Centroid[i],new_Centroid[i]) for i in range(self.k)]
        
        return sum(distances) == 0

    
    def func_create_clusters(self,centroid_data):
        create_clusters = [[] for j in range(self.k)]

        for i,y in enumerate(self.arr_data):
            index_centroid = self.func_closest_centroid(y,centroid_data)
            create_clusters[index_centroid].append(i)
        return create_clusters

    def func_closest_centroid(self, index_sample,centroid_data):

         # The distances calculation calling functions are as follows :--
        distance_btw_point = [euclidean_dist(index_sample,point_i) for point_i in centroid_data]
        #distance_btw_point = [cosine_dist(index_sample,point_i) for point_i in centroid_data]
        #distance_btw_point = [Jaccard_dist(index_sample,point_i) for point_i in centroid_data]
        
        closest_point = np.argmin(distance_btw_point)
        return closest_point


    def func_Getcentroids(self, getClusters):
        centroids = np.zeros((self.k,self.num_features))
        for index_i,cluster_y in enumerate(getClusters):
            cluster_mean = np.mean(self.arr_data[cluster_y],axis=0) 
            centroids[index_i] = cluster_mean 
        return centroids


def main():
    obj_dataset = pd.read_csv("data.csv")
    obj_testdata = pd.read_csv("label.csv")
    arr_dataset = obj_dataset.values
    arr_testdata = obj_testdata.values
    print(arr_dataset.shape)    
    print(arr_testdata.shape)
    obj_Kmeans = KMeans_Class(10,10)

    prediction = obj_Kmeans.Func_create_centroid(arr_dataset)

    print(prediction)
    print(prediction.shape)

    count = 0

    for i in range(len(prediction)):
        if(prediction[i] == arr_testdata[i][0]):
            count = count + 1
    
    print(count)
    print ((count/len(prediction))*100)

    print(obj_Kmeans.SSE_Val_List)

    
    
if __name__ == "__main__":
	main()










