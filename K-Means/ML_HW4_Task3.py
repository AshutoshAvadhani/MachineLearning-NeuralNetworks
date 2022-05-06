import numpy as np
from numpy import dot
from numpy.linalg import norm

def euclidean_dist(x1,x2):
    dist = 0
    for i,j in (x1,x2):
        dist += pow((i-j),2)
    return np.sqrt(dist)

def main():
    dataset_red = [(4.7,3.2),(4.9,3.1),(5.0,3.0),(4.6,2.9)]
    dataset_blue = [(5.9,3.2),(6.7,3.1),(6.0,3.0),(6.2,2.8)]

    dist = 0
    dist_arr = []

    for i in dataset_red:
        for j in dataset_blue:
            dist_arr.append([euclidean_dist(i,j),[i,j]])
            dist += dist_arr[-1][0]


    dist_arr_final = []

    for val in dist_arr:
        dist_arr_final.append(float(val[0]))


    dist_arr_final.sort()

    print("Closest points distance is --> "+format(dist_arr_final[0],".4f"))
    print("Farthest points distance is --> "+format(dist_arr_final[-1],".4f"))
    print("Average distance of all points is :--> " + format(dist/len(dist_arr_final),".4f") + " ")

if __name__ == "__main__":
	main()