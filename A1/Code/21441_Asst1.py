import os
import pickle
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Union


if not os.path.exists('../models'):
    os.makedirs('../models')
if not os.path.exists('../plots'):
    os.makedirs('../plots')


class DLModel:
    """
        Model Class to approximate the Z function as defined in the assignment.
    """

    def __init__(self):
        """Initialize the model."""
        self.Z0 = [None] * 10
        self.L = None
    
    def get_predictions(self, X, Z_0=None, w=10, L=None) -> np.ndarray:
        """Get the predictions for the given X values.

        Args:
            X (np.array): Array of overs remaining values.
            Z_0 (float, optional): Z_0 as defined in the assignment.
                                   Defaults to None.
            w (int, optional): Wickets in hand.
                               Defaults to 10.
            L (float, optional): L as defined in the assignment.
                                 Defaults to None.

        Returns:
            np.array: Predicted score possible
        """

        Z = Z_0 * (1 - np.exp(-L * X / Z_0))
        return Z
        

    def calculate_loss(self, Params, X, Y, w=10) -> float:
        """ Calculate the loss for the given parameters and datapoints.
        Args:
            Params (list): List of parameters to be optimized.
            X (np.array): Array of overs remaining values.
            Y (np.array): Array of actual average score values.
            w (int, optional): Wickets in hand.
                               Defaults to 10.

        Returns:
            float: Mean Squared Error Loss for the model parameters 
                   over the given datapoints.
        """
        Z_0 = Params[:-1]
        L = Params[-1]
        loss = 0
        for i in range(1,11):
            Y_pred = self.get_predictions(X[w == i],Z_0[i-1],L = L)
            loss += np.sum((Y[w==i]-Y_pred)**2)
        return loss
    
    def save(self, path):
        """Save the model to the given path.

        Args:
            path (str): Location to save the model.
        """
        with open(path, 'wb') as f:
            pickle.dump((self.L, self.Z0), f)
    
    def load(self, path):
        """Load the model from the given path.

        Args:
            path (str): Location to load the model.
        """
        with open(path, 'rb') as f:
            (self.L, self.Z0) = pickle.load(f)


def get_data(data_path) -> Union[pd.DataFrame, np.ndarray]:
    """
    Loads the data from the given path and returns a pandas dataframe.

    Args:
        path (str): Path to the data file.

    Returns:
        pd.DataFrame, np.ndarray: Data Structure containing the loaded data
    """
    data = pd.read_csv(data_path)
    return data

def convert_date(date_str):
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    if len(date_str.split(" ")) == 3:
        parts = date_str.split(" ")
        month = parts[0]
        month = str(months.index(month) + 1).zfill(2)
        date = parts[1].split("-")[0].zfill(2)
        year = parts[2]
    else:
        parts = date_str.split("/")
        month = parts[1]
        date = parts[0].zfill(2)
        year = parts[2]

    return f"{date}-{month}-{year}"


def preprocess_data(data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
    """Preprocesses the dataframe by
    (i)   removing the unnecessary columns,
    (ii)  loading date in proper format DD-MM-YYYY,
    (iii) removing the rows with missing values,
    (iv)  anything else you feel is required for training your model.

    Args:
        data (pd.DataFrame, nd.ndarray): Pandas dataframe containing the loaded data

    Returns:
        pd.DataFrame, np.ndarray: Datastructure containing the cleaned data.
    """
    # Converting date to required format
    data['Date'] = data['Date'].apply(convert_date)

    # Filtering first Innings Data 
    data = data[data['Innings'] == 1]

    #Removed rows with error
    data = data[data['Error.In.Data'] == 0]

    # Extracting required Columns
    data = data.loc[:,['Over','Runs','Innings.Total.Runs','Runs.Remaining','Wickets.in.Hand']]

    return data

def get_data_matrix(data):

    data_matrix = []
    for i in range(1,11):
        u = 50 - data[data['Wickets.in.Hand'] == i]['Over'].values
        z = data[data['Wickets.in.Hand'] == i]['Runs.Remaining'].values
        w = np.ones(len(u))*i
        data_matrix.extend(np.column_stack((u,z,w)))
    unique_matches = data[data['Over'] == 1]
    u = np.ones(len(unique_matches))*50
    z = unique_matches['Innings.Total.Runs'].values
    w = np.ones(len(unique_matches))*10 
    data_matrix.extend(np.column_stack((u, z, w)))
    return np.array(data_matrix)

def train_model(data: Union[pd.DataFrame, np.ndarray], model: DLModel) -> DLModel:
    """Trains the model

    Args:
        data (pd.DataFrame, np.ndarray): Datastructure containg the cleaned data
        model (DLModel): Model to be trained
    """
    data_matrix = get_data_matrix(data)
    parameters = [10, 30, 40, 60, 90, 125, 150, 170, 190, 200,10]
    optimal_solution = sp.optimize.minimize(model.calculate_loss,parameters,args = (data_matrix[:,0],data_matrix[:,1],data_matrix[:,2].astype(int)),method='L-BFGS-B')
    optimal_loss = optimal_solution['fun']
    optimal_parameters = optimal_solution['x']
    model.Z0 = list(optimal_parameters[:-1])
    model.L = optimal_parameters[-1]
    #model.normalized_loss = optimal_loss/len(data_matrix)

    return model


def plot(model: DLModel, plot_path: str) -> None:
    """ Plots the model predictions against the number of overs
        remaining according to wickets in hand.

    Args:
        model (DLModel): Trained model
        plot_path (str): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.title("Expected Runs vs Overs Remaining")
    plt.xlabel('Overs Remaining')
    plt.ylabel('Expected Runs')
    plt.grid(True)

    colors = plt.cm.tab10(np.linspace(0, 1, len(model.Z0)))

    x = np.linspace(0, 50, 100)
    for i, ans_value in enumerate(model.Z0):
        y_run = ans_value * (1 - np.exp(-model.L * x / ans_value))
        plt.plot(x, y_run, color=colors[i], label='Z[' + str(i + 1) + ']')

    plt.xlim((0, 50))
    plt.ylim((0, 250))
    plt.xticks([0, 10, 20, 30, 40, 50])
    plt.yticks([0, 50, 100, 150, 200, 250])

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(plot_path)

def print_model_params(model: DLModel) -> List[float]:
    '''
    Prints the 11 (Z_0(1), ..., Z_0(10), L) model parameters

    Args:
        model (DLModel): Trained model
    
    Returns:
        array: 11 model parameters (Z_0(1), ..., Z_0(10), L)

    '''
    parameters = model.Z0
    parameters.append(model.L)
    print(parameters)
    #return parameters


def calculate_loss(model: DLModel, data: Union[pd.DataFrame, np.ndarray]) -> float:
    '''
    Calculates the normalised squared error loss for the given model and data

    Args:
        model (DLModel): Trained model
        data (pd.DataFrame or np.ndarray): Data to calculate the loss on
    
    Returns:
        float: Normalised squared error loss for the given model and data
    '''
    data_matrix = get_data_matrix(data)
    parameters = model.Z0
    parameters.append(model.L)
    normalized_loss = model.calculate_loss(parameters,data_matrix[:,0],data_matrix[:,1],data_matrix[:,2].astype(int))/len(data_matrix)

    print(normalized_loss)
    #return  normalized_loss


def main(args):
    """Main Function"""

    data = get_data(args['data_path'])  # Loading the data
    print("Data loaded.")
    
    # Preprocess the data
    data = preprocess_data(data)
    print("Data preprocessed.")
    
    model = DLModel()  # Initializing the model
    model = train_model(data, model)  # Training the model
    model.save(args['model_path'])  # Saving the model
    
    plot(model, args['plot_path'])  # Plotting the model
    
    # Printing the model parameters
    print_model_params(model)

    # Calculate the normalised squared error
    calculate_loss(model, data)


if __name__ == '__main__':
    args = {
        "data_path": "../data/04_cricket_1999to2011.csv",
        "model_path": "../models/model.pkl",  # ensure that the path exists
        "plot_path": "../plots/plot.png",  # ensure that the path exists
    }
    main(args)
