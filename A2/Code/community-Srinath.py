import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import seaborn as sns
import time

def import_facebook_data(path):
    # Import text file with entries as int data type
    data = np.loadtxt(path).astype(int)
    return data

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

def compute_metrics(G,partition1,partition2):
    # Calculate cut
    cut = nx.cut_size(G, partition1, partition2)

    # Computing Ratio Cut
    R1 = cut/len(partition1) if len(partition1) > 0 else 0
    R2 = cut/len(partition2) if len(partition2) > 0 else 0
    Ratio_cut = R1 + R2

    # Calculate degree of partition1 and partition2
    degree_partition1 = sum(G.degree[node] for node in partition1)
    degree_partition2 = sum(G.degree[node] for node in partition2)

    # Computing Normalized Cut
    N1 = cut/degree_partition1 if degree_partition1 > 0 else 0
    N2 = cut/degree_partition2 if degree_partition2 > 0 else 0
    Normalized_cut = N1 + N2

    # Compute Conductance
    Conductance = cut/min(degree_partition1,degree_partition1) if cut > 0 else 0

    # Modularity
    Modularity = nx.community.modularity(G,[partition1,partition2])

    print("Ratio Cut: ", Ratio_cut)
    print("Normalized Cut: ", Normalized_cut)
    print("Conductance: ", Conductance)
    print("Modulaity: ", Modularity)

def plot_fiedler_vector(fiedler_vector,title):
    plt.figure(figsize=(12, 8))
    plt.scatter(np.arange(len(fiedler_vector)),np.sort(fiedler_vector, axis = 0).reshape(-1))
    plt.title(title)
    #plt.savefig("Sorted Fiedler Vector: " + title + '.png')
    plt.show()

def plot_adj_mat(A, fielder_vec, title):
    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Subplot 1: Original Adjacency Matrix
    sns.heatmap(A,ax=axes[0], cbar=False,xticklabels=False, yticklabels=False)
    axes[0].set_title("Original Adjacency Matrix: " + title)

    # Subplot 2: Adjacency Matrix Sorted by Fiedler Vector
    sorted_indices = np.argsort(fielder_vec, axis=0).reshape(-1)
    sorted_adj_mat_fb = A[sorted_indices][:, sorted_indices]
    sns.heatmap(sorted_adj_mat_fb, ax=axes[1], cbar=False,xticklabels=False, yticklabels=False)
    axes[1].set_title("Adjacency Matrix Sorted by Fiedler Vector: " + title)

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()
    #plt.savefig("Adjacency Matrix" + title + '.png')

def draw_graph(graph_partition,edges,title):
    G = nx.Graph()
    G.add_edges_from(edges)
    community_colors = graph_partition[:, 1] 
    pos = nx.spring_layout(G)
    nx.draw(G, pos, node_color=community_colors, cmap = plt.cm.Set1,with_labels=False)
    plt.title(title)
    plt.show()
    #plt.savefig("Graph: "+ title + '.png')
    

def spectralDecomp_OneIter(edges,p = True):

    # Create a NetworkX graph from the list of edges
    G = nx.Graph()  # Create an undirected graph
    G.add_edges_from(edges)
    true_node_id = np.array((list(G.nodes)))
    #true_node_id = np.array(sorted(G.nodes()))

    # Generate the adjacency matrix from the graph
    #adjacency_matrix = nx.to_numpy_array(G, nodelist=true_node_id ,dtype=float)
    adjacency_matrix = nx.to_numpy_array(G ,dtype=float)

    # Calculate degree matrix
    degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))

    # Calculate the Laplacian matrix
    laplacian_matrix = degree_matrix - adjacency_matrix
 
    #Calculate the eigenvalues and eigenvectors of the Laplacian matrix
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
        compute_metrics(G,set(true_node_id[pos_idx]),set(true_node_id[neg_idx]))
    
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
        fiedler_vector,_,graph_partition = spectralDecomp_OneIter(edges,p = False)
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
            if len(p1) > 50 and len(partition1_edges) > 0:
                self.recursiveSpectralDecomposition(partition1_edges)
            if len(p2) > 50 and len(partition2_edges) > 0:
                self.recursiveSpectralDecomposition(partition2_edges)
    
def spectralDecomposition(edges):
    start_time = time.time()
    S = Spectral_Decompostion(edges)
    S.recursiveSpectralDecomposition(edges)
    end_time = time.time()
    elapsed_time = end_time - start_time
    graph_partition = np.array(list(S.community_dict.items()))
    num_communities,count = np.unique(graph_partition[:,1],return_counts=True)
    print("No of Communities Formed:",len(num_communities))
    print("Sizes of Communities:",count)
    print("Final Modularity:", S.final_modularity(edges,graph_partition))
    print(f"Time taken: {elapsed_time} seconds")
    return graph_partition

def createSortedAdjMat(graph_partition, edges,title):
    true_node_id = np.unique(edges).reshape(-1)
    adjacency_matrix = np.zeros((len(true_node_id), len(true_node_id))).astype(int)
    for e in edges:
        adjacency_matrix[e[0], e[1]] = 1
        adjacency_matrix[e[1], e[0]] = 1
    indices = graph_partition[:,0][np.argsort(graph_partition[:,1])]
    adjacency_matrix = np.take(adjacency_matrix, indices, axis = 0)
    adjacency_matrix = np.take(adjacency_matrix, indices, axis = 1)
    # Create a Seaborn heatmap
    plt.figure(figsize=(8, 8))
    sns.heatmap(adjacency_matrix,cbar=False,xticklabels=False, yticklabels=False)

    plt.title("Adjacency Matrix Sorted by Community IDs" + title)
    plt.show()
    return adjacency_matrix

class Louvain:
    def __init__(self,edges) -> None:
        self.G = nx.Graph()
        self.G.add_edges_from(edges)
        self.n = self.G.number_of_nodes()
        self.A_norm = nx.to_numpy_array(self.G, dtype=int)/(2*self.G.number_of_edges())
        self.degree = np.sum(self.A_norm,axis = 1)
        self.compute_neighbours()
        self.communities = np.arange(self.n)

    def compute_neighbours(self):
        self.neighbours = []
        for i in range(self.n):
            self.neighbours.append(np.where(self.A_norm[i]!=0)[0])

    def compute_Q_merge(self,i,C):
        communityC_nodes = np.where(self.communities == C)[0]
        sigma_total = sum(self.degree[node] for node in communityC_nodes)
        k_i_in = 2*np.sum(self.A_norm[i, communityC_nodes])
        k_i = self.degree[i]
        Q_merge = k_i_in - 2*sigma_total*k_i
        return Q_merge

    def compute_demerge(self,i):
        C = self.communities[i]
        communityC_nodes = np.where(self.communities == C)[0]
        sigma_total = sum(self.degree[node] for node in communityC_nodes)
        k_i_out = 2*np.sum(self.A_norm[i, communityC_nodes])
        k_i = self.degree[i]
        Q_demerge = 2*k_i*sigma_total - 2*k_i**2 - k_i_out
        return Q_demerge

    def compute_modularity(self):
        community_id = np.unique(self.communities)
        Q = 0
        for C in community_id:
            communityC_nodes = np.where(self.communities == C)[0]
            sigma_total = sum(self.degree[node] for node in communityC_nodes)
            sigma_in = np.sum(self.A_norm[np.ix_(communityC_nodes,communityC_nodes)])
            Q += sigma_in - sigma_total**2
        return Q

    def phase1(self):
        while True:
            count = 0
            for i in range(self.n):
                neighbour_communities = np.unique(self.communities[self.neighbours[i]])
                Q_demerge = self.compute_demerge(i)
                Q_max = 0
                best_community = self.communities[i]
                for j in neighbour_communities:
                    if j == self.communities[i]:
                        continue
                    Q_merge = self.compute_Q_merge(i,j)
                    delta_Q = Q_demerge + Q_merge
                    if delta_Q > Q_max:
                        Q_max = delta_Q
                        best_community = j

                if Q_max > 0 and best_community != self.communities[i]:
                    self.communities[i] = best_community
                    count += 1
            if count == 0:
                break

def louvain_one_iter(edges):
    start_time = time.time()
    L =  Louvain(edges)
    L.phase1()
    end_time = time.time()
    elapsed_time = end_time - start_time
    node_id = np.array(L.G.nodes).astype(int)
    community_id = L.communities
    graph_partition = np.column_stack([node_id,community_id])
    graph_partition2 = graph_partition.copy()
    for i in set(graph_partition[:,1]):
        idx = np.where(graph_partition[:,1] == i)[0]
        id = np.min(graph_partition[:,0][idx])
        graph_partition2[:,1][idx] = id
    print("Number of Communities formed: ", len(np.unique(L.communities)))
    print("Final Modularity: ", L.compute_modularity())
    print(f"Time taken: {elapsed_time} seconds")
    return graph_partition2
    
if __name__ == "__main__":

######################### Facebook data #########################

    print("######################### Facebook Dataset #########################")

    # Import facebook_combined.txt
    nodes_connectivity_list_fb = import_facebook_data("../data/facebook_combined.txt")
    # Question no. 1
    print("\n")
    print("____________Question 1: Spectral Decompostion One Iteration____________")
    fielder_vec_fb, adj_mat_fb, graph_partition_fb = spectralDecomp_OneIter(nodes_connectivity_list_fb)
    # plot_fiedler_vector(fielder_vec_fb, "Sorted Fiedler Vector (MinCut): Facebook")
    # plot_adj_mat(adj_mat_fb,fielder_vec_fb,"Facebook")
    # draw_graph(graph_partition_fb,nodes_connectivity_list_fb,'Graph: Spectral Decompsition One Iter Facebook')

    # # Question 2
    print("\n")
    print("____________Question 2: Spectral Decompostion Multiple Iterations____________")
    graph_partition_fb2 = spectralDecomposition(nodes_connectivity_list_fb)
    # draw_graph(graph_partition_fb2,nodes_connectivity_list_fb,'Graph: Spectral Decompsition Facebook')

    # Question 3
    # clustered_adj_mat_btc = createSortedAdjMat(graph_partition_fb2, nodes_connectivity_list_fb,"Facebook")

    # This is for question no. 4
    print("\n")
    print("____________Question 4: Louvain Algorithm____________")
    graph_partition_louvain_fb = louvain_one_iter(nodes_connectivity_list_fb)
    # # clustered_adj_mat_louvain_fb = createSortedAdjMat(graph_partition_louvain_fb, nodes_connectivity_list_fb," Louvain - Facebook")
    # draw_graph(graph_partition_louvain_fb,nodes_connectivity_list_fb,'Graph: Louvain Facebook')

 ###################### Bitcoin data ####################

    print("\n")
    print("######################### Bitcoin Dataset #########################")

    # Import soc-sign-bitcoinotc.csv
    nodes_connectivity_list_btc = import_bitcoin_data("../data/soc-sign-bitcoinotc.csv")

    # # Question 1
    print("\n")
    print("____________Question 1: Spectral Decompostion One Iteration____________")
    fielder_vec_btc, adj_mat_btc, graph_partition_btc = spectralDecomp_OneIter(nodes_connectivity_list_btc)
    # plot_fiedler_vector(fielder_vec_btc, "Sorted Fiedler Vector (MinCut): Bitcoin")
    # plot_adj_mat(adj_mat_btc,fielder_vec_btc,"Bitcoin")
    # draw_graph(graph_partition_btc,nodes_connectivity_list_btc,'Graph: Spectral Decompsition One Iter Bitcoin')

    # # Question 2
    print("\n")
    print("____________Question 2: Spectral Decompostion Multiple Iterations____________")
    graph_partition_btc2 = spectralDecomposition(nodes_connectivity_list_btc)
    # # draw_graph(graph_partition_btc2,nodes_connectivity_list_btc,'Graph: Spectral Decompsition Bitcoin')

    # # Question 3
    # clustered_adj_mat_btc = createSortedAdjMat(graph_partition_btc2, nodes_connectivity_list_btc,"Bitcoin")

    # # Question 4
    print("\n")
    print("____________Question 4: Louvain Algorithm____________")
    graph_partition_louvain_btc = louvain_one_iter(nodes_connectivity_list_btc)
    # clustered_adj_mat_louvain_btc = createSortedAdjMat(graph_partition_louvain_btc, nodes_connectivity_list_btc," Louvain - Bitcoin")
    # draw_graph(graph_partition_louvain_btc,nodes_connectivity_list_btc,'Graph: Louvain Bitcoin')