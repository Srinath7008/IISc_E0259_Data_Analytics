import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import seaborn as sns

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
    plt.plot(np.arange(len(fiedler_vector)),np.sort(fiedler_vector, axis = 0).reshape(-1))
    plt.title("Sorted Fiedler Vector: " + title)
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
    plt.savefig("Adjacency Matrix" + title + '.png')

def generate_colors(community_ids):
    color_mapping = {}
    for community_id, color in zip(set(community_ids),plt.get_cmap('viridis',len(set(community_ids)))(np.arange(len(set(community_ids))))):
        color_mapping[community_id]=color
    community_colors = [color_mapping[i] for i in community_ids]
    return community_colors

def draw_graph(graph_partition,edges,title):
    G = nx.Graph()
    G.add_edges_from(edges)
    community_colors = generate_colors(graph_partition[:, 1])
    pos = nx.spring_layout(G)
    nx.draw(G, pos, node_color=community_colors, with_labels=False)
    plt.title("Graph: " + title)
    plt.show()
    #plt.savefig("Graph: "+ title + '.png')

def spectralDecomp_OneIter(edges,criteria,p = True):

    # Create a NetworkX graph from the list of edges
    G = nx.Graph()  # Create an undirected graph
    G.add_edges_from(edges)
    true_node_id = np.array(sorted(list(G.nodes)))

    # Generate the adjacency matrix from the graph
    adjacency_matrix = nx.to_numpy_array(G, dtype=float)

    # Calculate degree matrix
    degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))

    # Calculate the Laplacian matrix
    laplacian_matrix = degree_matrix - adjacency_matrix

    # Calculate the 2nd smallest eigencvalue and fiedler vector of the Laplacian matrix
    if criteria == 'Min_cut':
        _, fiedler_vector = eigh(laplacian_matrix, subset_by_index=[1, 1])
    else:
        _, fiedler_vector = eigh(laplacian_matrix, degree_matrix, subset_by_index=[1, 1])

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
    L =  Louvain(edges)
    L.phase1()
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
    return graph_partition2

if __name__ == "__main__":

   ######################### Facebook data #########################
    print("Facebook Dataset:")
    print("No of nodes: 4039")
    print("No of edges: 88234")
    print("\n")

    # # Import facebook_combined.txt
    nodes_connectivity_list_fb = import_facebook_data("../data/facebook_combined.txt")

    # # Question no. 1
    # # Critertia Mincut
    # print("Critertia: MinCut: ")
    # fielder_vec_fb_m, adj_mat_fb_m, graph_partition_fb_m = spectralDecomp_OneIter(nodes_connectivity_list_fb,"Mincut")
    # plot_fiedler_vector(fielder_vec_fb_m, "Min Cut - Facebook Data")
    # print("plot saved")
    # plot_adj_mat(adj_mat_fb_m,fielder_vec_fb_m,"MinCut - Facebook Data")
    # print("plot saved")
    # draw_graph(graph_partition_fb_m,nodes_connectivity_list_fb,'Mincut - Facebook')
    # print("plot saved")


    # This is for question no. 2. Use the function 
    #graph_partition_fb = spectralDecomposition(nodes_connectivity_list_fb)
    #draw_graph(graph_partition_fb,nodes_connectivity_list_fb,'Spectral - Facebook')
    # # This is for question no. 3
    # # Create the sorted adjacency matrix of the entire graph. You will need the identified communities from
    # # question 3 (in the form of the nx2 numpy array graph_partition) and the nodes_connectivity_list. The
    # # adjacency matrix is to be sorted in an increasing order of communitites.
    # clustered_adj_mat_fb = createSortedAdjMat(graph_partition_fb, nodes_connectivity_list_fb)

    # # This is for question no. 4
    # print("Louvain Algorithm for Facebook Data")
    # graph_partition_louvain_fb = louvain_one_iter(nodes_connectivity_list_fb)
    # draw_graph(graph_partition_louvain_fb,nodes_connectivity_list_fb,'Louvain - Facebook')

    ###################### Bitcoin data ####################

    # print("Bitcoin Dataset:")
    # print("No of nodes: 5881")
    # print("No of edges: 21492")
    # print("\n")
    # Import soc-sign-bitcoinotc.csv
    # nodes_connectivity_list_btc = import_bitcoin_data("../data/soc-sign-bitcoinotc.csv")

    # Question 1
    # Critertia Mincut
    # print("Critertia: MinCut: ")
    # fielder_vec_btc_m, adj_mat_btc_m, graph_partition_btc_m = spectralDecomp_OneIter(nodes_connectivity_list_btc,"Min_cut")
    # plot_fiedler_vector(fielder_vec_btc_m, "Min Cut - Bitcoin Data")
    # plot_adj_mat(adj_mat_btc_m,fielder_vec_btc_m,"MinCut - Bitcoin Data")
    # draw_graph(graph_partition_btc_m,nodes_connectivity_list_btc,'MinCut - BitCoin')

    # # Question 2
    # graph_partition_btc = spectralDecomposition(nodes_connectivity_list_btc)

    # # Question 3
    # clustered_adj_mat_btc = createSortedAdjMat(graph_partition_btc, nodes_connectivity_list_btc)

    # # Question 4
    # print("Louvain Algorithm for Bitcoin Data")
    # graph_partition_louvain_btc = louvain_one_iter(nodes_connectivity_list_btc)
    # draw_graph(graph_partition_louvain_btc,nodes_connectivity_list_btc,'Louvain - BitCoin')
