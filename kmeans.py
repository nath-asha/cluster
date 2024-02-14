from linear_algebra import Vector
import matplotlib.pyplot as plt
def num_differences(v1: Vector, v2: Vector) -> int:
    assert len(v1) == len(v2)
    return len([x1 for x1, x2 in zip(v1, v2) if x1 != x2])

assert num_differences([1, 2, 3], [2, 1, 3]) == 2
assert num_differences([1, 2], [1, 2]) == 0

from typing import List
from linear_algebra import vector_mean

def cluster_means(k: int,
                  inputs: List[Vector],
                  assignments: List[int]) -> List[Vector]:
    # clusters[i] contains the inputs whose assignment is i
    clusters = [[] for i in range(k)]
    for input, assignment in zip(inputs, assignments):
        clusters[assignment].append(input)

    # if a cluster is empty, just use a random point
    return [vector_mean(cluster) if cluster else random.choice(inputs)
            for cluster in clusters]

import itertools
import random
import tqdm
from linear_algebra import squared_distance

class KMeans:
    def __init__(self, k: int) -> None:
        self.k = k                      # number of clusters
        self.means = None

    def classify(self, input: Vector) -> int:
        """return the index of the cluster closest to the input"""
        return min(range(self.k),
                   key=lambda i: squared_distance(input, self.means[i]))

    def train(self, inputs: List[Vector]) -> None:
        # Start with random assignments
        assignments = [random.randrange(self.k) for _ in inputs]

        with tqdm.tqdm(itertools.count()) as t:
            for _ in t:
                # Compute means and find new assignments
                self.means = cluster_means(self.k, inputs, assignments)
                new_assignments = [self.classify(input) for input in inputs]

                # Check how many assignments changed and if we're done
                num_changed = num_differences(assignments, new_assignments)
                if num_changed == 0:
                    return

                # Otherwise keep the new assignments, and compute new means
                assignments = new_assignments
                self.means = cluster_means(self.k, inputs, assignments)
                t.set_description(f"changed: {num_changed} / {len(inputs)}")
                


def main():
    
    
    inputs: List[List[float]] = [[-14,-5],[13,13],[20,23],[-19,-11],[-9,-16],[21,27],[-49,15],[26,13],[-46,5],[-34,-1],[11,15],[-49,0],[-22,-16],[19,28],[-12,-8],[-13,-19],[-41,8],[-11,-6],[-25,-9],[-18,-3]]
    x=[-14,13,20,-19,-9,21,-49,26,-46,-34,11,-49,-22,19,-12,-13,-41,-11,-25,-18]
    y=[[-14,-5],[13,13],[20,23],[-19,-11],[-9,-16],[21,27],[-49,15],[26,13],[-46,5],[-34,-1],[11,15],[-49,0],[-22,-16],[19,28],[-12,-8],[-13,-19],[-41,8],[-11,-6],[-25,-9],[-18,-3]]
    x=[-14,13,20,-19,-9,21,-49,26,-46,-34,11,-49,-22,19,-12,-13,-41,-11,-25,-18]]
    plt.scatter(inputs,inputs,marker='.',color='green',label='i')
    
    random.seed(12)                   # so you get the same results as me
    clusterer = KMeans(k=3)
    clusterer.train(inputs)
    means = sorted(clusterer.means)   # sort for the unit test
    
    assert len(means) == 3
    
    # Check that the means are close to what we expect.
    assert squared_distance(means[0], [-44, 5]) < 1
    assert squared_distance(means[1], [-16, -10]) < 1
    assert squared_distance(means[2], [18, 20]) < 1
    
    """random.seed(0)
    clusterer = KMeans(k=2)
    clusterer.train(inputs)
    means = sorted(clusterer.means)
    
    assert len(means) == 2
    assert squared_distance(means[0], [-26, -5]) < 1
    assert squared_distance(means[1], [18, 20]) < 1 """
    
    from matplotlib import pyplot as plt
    
    def squared_clustering_errors(inputs: List[Vector], k: int) -> float:
        """finds the total squared error from k-means clustering the inputs"""
        clusterer = KMeans(k)
        clusterer.train(inputs)
        means = clusterer.means
        assignments = [clusterer.classify(input) for input in inputs]
    
        return sum(squared_distance(input, means[cluster])
                   for input, cluster in zip(inputs, assignments))
    
    # now plot from 1 up to len(inputs) clusters
    
    ks = range(1, len(inputs) + 1)
    errors = [squared_clustering_errors(inputs, k) for k in ks]
    
    plt.plot(ks, errors)
    plt.xticks(ks)
    plt.xlabel("k")
    plt.ylabel("total squared error")
    plt.title("Total Error vs. # of Clusters")
    # plt.show()
    
    
    

if __name__ == "__main__": main()
