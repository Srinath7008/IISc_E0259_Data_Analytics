import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import seaborn as sns

def import_bitcoin_data(path):
    data = np. genfromtxt(path, delimiter=',')
    data = data[:,[0,1]].astype(int)
    unique_edges = set()
    unique_edges_list = []
    for e in data:
        e = tuple(sorted(e))
        if e not in unique_edges:
            unique_edges.add(e)
            unique_edges_list.append(e)  
    true_node_ids = np.unique(unique_edges_list).reshape(-1)
    node_mapping = {true_node_id: new_node_id for new_node_id,true_node_id in enumerate(true_node_ids)}
    inv_node_mapping = {new_node_id: true_node_id for new_node_id,true_node_id in enumerate(true_node_ids)}
    unique_edges = np.array(unique_edges_list)
    for i in range(len(unique_edges)):
        for j in range(2):
            unique_edges[i][j] = node_mapping[unique_edges[i][j]]
    return unique_edges

def import_facebook_data(path):
    # Import text file with entries as int data type
    data = np.loadtxt(path).astype(int)
    return data

def spectralDecomp_OneIter(edges,criteria,p = True):

    # Create a NetworkX graph from the list of edges
    G = nx.Graph()  # Create an undirected graph
    G.add_edges_from(edges)
    true_node_id = np.array((list(G.nodes)))

    # Generate the adjacency matrix from the graph
    adjacency_matrix = nx.to_numpy_array(G, dtype=float)

    # Calculate degree matrix
    degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))

    # Calculate the Laplacian matrix
    laplacian_matrix = degree_matrix - adjacency_matrix

    # # Calculate the 2nd smallest eigencvalue and fiedler vector of the Laplacian matrix
    # if criteria == 'Min_cut':
    #     _, fiedler_vector = eigh(laplacian_matrix, subset_by_index=[1, 1])
    # else:
    #     _, fiedler_vector = eigh(laplacian_matrix, degree_matrix, subset_by_index=[1, 1])

    
    # Calculate the eigenvalues and eigenvectors of the Laplacian matrix
    eigenvalues, eigenvectors = eigh(laplacian_matrix)

    # Sort the eigenvalues and corresponding eigenvectors
    sorted_indices = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # The second smallest eigenvalue corresponds to the Fiedler vector
    fiedler_vector = eigenvectors[:, 1]

    community_ids = np.zeros(len(true_node_id))
    pos_idx = np.where(fiedler_vector >= 0)[0]
    neg_idx = np.where(fiedler_vector < 0)[0]
    
    if(len(pos_idx)) > 0:
        pos_id = np.min(true_node_id[pos_idx])
        community_ids[pos_idx] = pos_id
    if(len(neg_idx)) > 0:
        neg_id = np.min(true_node_id[neg_idx])
        community_ids[neg_idx] = neg_id

    graph_partition = np.column_stack([true_node_id,community_ids]).astype(int)

    if p:
        print("Communities formed: 2")
        print("Community 1 size: ",len(pos_idx))
        print("Community 2 size: ",len(neg_idx))
        #compute_metrics(G,set(true_node_id[pos_idx]),set(true_node_id[neg_idx]))
    
    return fiedler_vector, adjacency_matrix, graph_partition


class Spectral_Decompostion:
    def __init__(self,edges) -> None:
        self.true_node_id = np.unique(edges).reshape(-1)
        self.community_dict = {node_id:node_id for node_id in self.true_node_id}
        
    def stopping_criteria(self,fiedler_vector):
        fiedler_vector = np.sort(fiedler_vector)
        p = len(np.where(fiedler_vector >= 0)[0])
        n = len(fiedler_vector) - p
        diff = [fiedler_vector[i+1] - fiedler_vector[i] for i in range(len(fiedler_vector)-1)]
        # std_dev = np.std(fiedler_vector)
        std_dev = np.std(diff)
        threshold = 2 * std_dev
        if max(diff) < threshold or min(p,n) < 25:
            return True
        else:
            return False
        
    def final_modularity(self,edges,graph_partition):
        G = nx.Graph()
        G.add_edges_from(edges)
        unique_id = np.unique(graph_partition[:,1])
        communities = []
        for i in unique_id:
            C = set(graph_partition[:,0][np.where(graph_partition[:,1] == i)[0]])
            communities.append(C)
        Q = nx.community.modularity(G, communities)
        return Q

    def recursiveSpectralDecomposition(self,edges):
        # Apply spectral decomposition for one iteration
        fiedler_vector,_,graph_partition = spectralDecomp_OneIter(edges,"Min_cut",p = False)
        if self.stopping_criteria(fiedler_vector):
            return

        for i in range(len(graph_partition)):
            self.community_dict[graph_partition[i][0]] = graph_partition[i][1]

        unique_c = np.unique(graph_partition[:,1])
        if len(unique_c) == 2:
            p1 = graph_partition[:,0][np.where(graph_partition[:,1] == unique_c[0])[0]]
            p2 = graph_partition[:,0][np.where(graph_partition[:,1] == unique_c[1])[0]]
            partition1_edges = [(u, v) for u, v in edges if u in p1 and v in p1]
            partition2_edges = [(u, v) for u, v in edges if u in p2 and v in p2]
            if len(p1) > 100 and len(partition1_edges) > 0:
                self.recursiveSpectralDecomposition(partition1_edges)
            if len(p2) > 100 and len(partition2_edges) > 0:
                self.recursiveSpectralDecomposition(partition2_edges)

def spectralDecomposition(edges):
    S = Spectral_Decompostion(edges)
    S.recursiveSpectralDecomposition(edges)
    graph_partition = np.array(list(S.community_dict.items()))
    num_communities,count = np.unique(graph_partition[:,1],return_counts=True)
    print("No of Communities Formed:",len(num_communities))
    print("Sizes of Communities:",count)
    print("Final Modularity:", S.final_modularity(edges,graph_partition))
    return graph_partition

if __name__ == "__main__":
    e1 = import_bitcoin_data("../data/soc-sign-bitcoinotc.csv")
    e2 = import_facebook_data("../data/facebook_combined.txt")
    gp = spectralDecomposition(e1)
    gp2 = spectralDecomposition(e2)