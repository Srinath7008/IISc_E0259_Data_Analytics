import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f

if not os.path.exists('../plots'):
    os.makedirs('../plots')

class Two_way_ANOVA:
    def __init__(self,df):
        self.data = df.iloc[:,1:49].values
        N, D = self.calculate_N_and_D()
        self.mat_num = np.eye(48) - N @ np.linalg.pinv(N.T @ N) @ N.T
        self.mat_den = np.eye(48) - D @ np.linalg.pinv(D.T @ D) @ D.T

        self.compute_pvalues()
    
    def calculate_N_and_D(self):

        num_rows = 48
        N = np.zeros((num_rows, 4), dtype=int)
        D = np.zeros((num_rows, 4), dtype=int)

        # Define the column indices for each combination
        male_non_smoker = slice(0, 12)  # Male, Non-Smoker
        male_smoker = slice(12, 24)     # Male, Smoker
        female_non_smoker = slice(24, 36)  # Female, Non-Smoker
        female_smoker = slice(36, 48)     # Female, Smoker

        # Set values in matrix N for the corresponding combinations
        N[male_non_smoker, [0, 3]] = 1      # Male, Non-Smoker (columns 0 and 2)
        N[male_smoker, [0, 2]] = 1         # Male, Smoker (columns 0 and 3)
        N[female_non_smoker, [1, 3]] = 1   # Female, Non-Smoker (columns 1 and 2)
        N[female_smoker, [1, 2]] = 1       # Female, Smoker (columns 1 and 3)

        # Set values in matrix D for the corresponding combinations
        D[male_non_smoker, 1] = 1  # M-NS
        D[male_smoker, 0] = 1     # M-S
        D[female_non_smoker, 3] = 1  # F-NS
        D[female_smoker, 2] = 1      # F-S
    
        return N,D
    
    def pvalue_single_point(self,x):
        f_statistic_n = x.T @ self.mat_num @ x
        f_statistic_d = x.T @ self.mat_den @ x
        if f_statistic_d == 0:
            return None
        f_statistic = (f_statistic_n/f_statistic_d - 1) * 44 # 48(n) - 4(Rank of D) = 44, Rank(D) - Rank(N) = 1 
        cdf = f.cdf(f_statistic,1,44) # Computing F-Statistic at f(a,b) where a = Rank(D) - Rank(N) = 1 and b = n - Rank(D) = 44
        p_value = 1 - cdf
        return p_value
    
    def compute_pvalues(self):
        p_values = []
        for i in range(len(self.data)):
            p = self.pvalue_single_point(self.data[i].reshape(-1, 1))
            p_values.append(p)     
        self.filtered_p_values = [item[0][0] for item in p_values if item is not None]

    def plot_histogram(self,plot_path):
        plt.figure(figsize=(8, 6))  # Set the figure size

        plt.hist(self.filtered_p_values, bins=20, color='skyblue', edgecolor='black', alpha=0.7)  # Adjust bins and colors
        plt.xlabel('P-Values')
        plt.ylabel('Frequency')
        plt.title('Histogram of P-Values')

        # Add grid lines
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Add a vertical line at a specific value if needed
        plt.axvline(x=0.05, color='red', linestyle='--', label='Significance Threshold')
        # Add label next to the red line
        plt.text(0.055, 2500, 'p = 0.05', color='red', rotation=0)

        plt.tight_layout()  # Improve spacing
        plt.savefig(plot_path)
        print("Histogram saved.")
        #plt.show()

if __name__ == '__main__':
    file_path = "../data/Raw Data_GeneSpring.txt"
    plot_path = "../plots/histogram.png"

    # Read the text file into a DataFrame
    smoking_data = pd.read_csv(file_path, sep='\t')
    smoking_anova = Two_way_ANOVA(smoking_data)
    smoking_anova.plot_histogram(plot_path)