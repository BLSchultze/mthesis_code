"""# Functions to facilitate working with the *Drosophila* connectome dataset

This module contains the following functions:
    - fetch_all_connections:    download all connections from the MANC dataset
    - fetch_all_neuron_props:   download additional information from the MANC dataset for large sets of neurons
    - create_neuron_dict:       create a dictionary mapping neuron body IDs to neuron types or vice versa
    - flatten:                  flatten a list of lists
    - roi_info_summary:         sum up the counts of pre- and postsynaptic sites

For retrieving information from the MANC dataset a neuprint token is necessary. Please place it as a .txt file in 
"./accessory_files/neuprint_token.txt" relative to the directory the module is loaded from. 
    
Author:         Bjarne Schultze
Last modified:  02.12.2024
"""

import numpy as np
import pandas as pd
import neuprint as neup

# Try to load the neuprint auth token
try:
    # Save authentication token and create neuPrint+ client
    with open("./accessory_files/neuprint_token.txt") as f:
        _auth_token = f.readlines()[0]
    # Create MANC neuprint client
    neu_cl = neup.Client('neuprint.janelia.org', dataset='manc:v1.2.1', token=_auth_token)
except FileNotFoundError:
    # Set client variable to none
    neu_cl = None
    # Print message
    print("No neuprint auth token provided as './neuprint_token.txt'! Not all functions might be available.")



def fetch_all_connections(export:bool=True) -> pd.DataFrame|None:
    """### Download all connections from the MANC dataset via the neuprint interface
    The connections are stored in pandas DataFrames. Next to the bodyIDs and the and the connection weight (number of connections)
    the neuron types and instances are stored. The data can be written to a h5 file on request. 

    Args:
        export (bool): indicate whether to export the data to an pandas-readable h5 file (True) or not (False), default: True
    
    Returns:
        DataFrame|None: either none (if export=True) or a pandas DataFrame with the connection information    
    """
    # Error if no neuprint client
    if not isinstance(neu_cl, neup.Client):
        raise RuntimeError("No neuprint client defined due to a missing auth token.")
    
    # Download the connection information
    neuron_info, connections = neup.fetch_traced_adjacencies(client=neu_cl)
    # Add the type and instance information
    conn_table = neup.merge_neuron_properties(neuron_info, connections, ['type','instance'])

    # Export or return the dataset
    if export:
        conn_table.to_hdf("manc_1.2.1_connections.h5", key="connections")
    else:
        return conn_table



def fetch_all_neuron_props(conn_table_path:str="../additional_files/manc_1.2.1_connections.h5", export:bool=True) -> pd.DataFrame|None:
    """### Fetch neuron information for large sets of neurons via the neuprint interface
    All neuron information are stored in pandas DataFrames. The data can be written to a h5 file on request. 
    See following sites for further information: <br>
    https://neuprint.janelia.org/public/neuprintuserguide.pdf <br>
    https://www.biorxiv.org/content/10.1101/2020.01.16.909465v1.full.pdf
    
    Following information are stored: 
    'bodyId', 'instance', 'type', 'pre', 'post', 'downstream',
    'upstream', 'size', 'status', 'statusLabel', 'somaLocation', 'roiInfo', 'target', 'somaNeuromere', 'serial', 'synonyms', 'rootSide',
    'celltypeTotalNtPredictions', 'hemilineage', 'entryNerve', 'location', 'ntUnknownProb', 'predictedNt', 'systematicType', 'predictedNtProb',
    'tosomaLocation', 'celltypePredictedNt', 'transmission', 'birthtime', 'group', 'locationType', 'rootLocation', 'synweight', 'ntGabaProb',
    'serialMotif', 'avgLocation', 'subclassabbr', 'prefix', 'receptorType', 'description', 'source', 'ntAcetylcholineProb', 'origin', 'subcluster',
    'ntGlutamateProb', 'subclass', 'tag', 'cluster', 'exitNerve', 'modality', 'somaSide', 'totalNtPredictions', 'vfbId', 'longTract',
    'class', 'inputRois', 'outputRois'
    
    Args:
        conn_table_path (str): path to a pandas-readable h5 file holding connection information about the neurons for which to
                               fetch additional information
        export (bool): indicate whether to export the data to an pandas-readable h5 file (True) or not (False), default: True
    
    Returns:
        DataFrame|None: either none (if export=True) or a pandas DataFrame holding the information for the neurons in the connections table    
    """
    # Error if no neuprint client
    if not isinstance(neu_cl, neup.Client):
        raise RuntimeError("No neuprint client defined due to a missing auth token.")
    
    # Load the connections table from given path
    conn_table = pd.read_hdf(conn_table_path)
    # Extract the bodyIDs 
    bodyIds = pd.unique(conn_table.loc[:, 'bodyId_pre'])

    # Create indices to chunk the bodyIds to prevent request restrictions
    indices = np.arange(0, bodyIds.shape[0], 1200)
    indices = np.append(indices, bodyIds.shape[0]-1)

    # Allocate empty data frame to return if no data is found
    neuron_info = pd.DataFrame([])

    for i,j in zip(indices[:-1],indices[1:]):
        if i == 0:
            # Fetch the neuron information
            neuron_info, _ = neup.fetch_neurons(bodyIds[i:j], client=neu_cl)
        else:
            # Fetch the neuron information
            n_info, _ = neup.fetch_neurons(bodyIds[i:j], client=neu_cl)
            # Append new neurons to the neuron data frame
            neuron_info = pd.concat([neuron_info,n_info], axis=0)

    # Reset the index of the data frame
    neuron_info = neuron_info.reset_index()

    # Export or return the dataset
    if export:
        neuron_info.to_hdf("./manc_1.2.1_neurons.h5", key="neurons")
    else:
        return neuron_info



def create_neuron_dict(neuron_table:pd.DataFrame, key_type:str='id') -> dict:
    """### Create a dictionary to quickly look up neuron types or IDs
    
    Args:
        neuron_table (DataFrame): a pandas DataFrame with at least the neuron body IDs and neuron types
        key_type (str): indicate which information to use as a key, either the body IDs (key_tpye="id") or the neuron type 
                        (key_type="type"), default: "id"
    
    Returns:
        dict: a dictionary mapping the body IDs to neuron types (key_type="type") or a dictionary mapping the neuron types to body IDs (key_type="id")
    """
    # For the neuron IDs as keys
    if key_type == 'id':
        # Add each neuron ID-type allocation to the dictionary
        neuron_dict = {nid:ntype for nid,ntype in zip(neuron_table['bodyId'],neuron_table['type'])}
    # For the neuron types as keys
    elif key_type == 'type':
        # Create empty dict
        neuron_dict = {}
        # Iterate over the neuron types
        for ntype in neuron_table['type'].unique():
            # Get the bodyIDs for all neurons of the same type
            bodyIds = np.array(neuron_table.loc[neuron_table['type']==ntype, 'bodyId'])
            # Assign the bodyIDs to the neuron type in the dict
            neuron_dict[ntype] = bodyIds
    else:
        # Return empty dict with message in all other cases
        neuron_dict = {}
        print("Empty dict! Please use either 'id' or 'type' as the key type.")

    return neuron_dict



def generate_conn_mat(connections:pd.DataFrame, row_grp:str, col_grp:str, subset:None|list[list|np.ndarray]=None, 
                      normalize:bool=True, weight_col:str="weight", return_norm_arr=False) -> tuple[pd.DataFrame, pd.Series]|pd.DataFrame:
    """### Generate a connectivity matrix from a connections data frame

    Args:
        connections (DataFrame): data frame listing all connections or a relevant subset
        row_grp (str): name of the column which should be used for grouping the data, later corresponding to the rows of the matrix
        col_grp (str): name of second column which should be used for grouping the data, later corresponding to the columns of the matrix
        subset (None, list): either none for the full connection data frame to be used or a list of two lists/arrays, first list/array with 
                             neurons to select from the 'row_grp' column, second list/array with neurons to select from the 'col_grp' column, 
                             default: None (all given data is used)
        normalize (bool): if True, the summed connection weights for each connection are divided by the total number of input connections the 
                          target neuron receives (this requires that 'connections' holds all connections for the neurons of interest), 
                          if False, the connectivity matrix holds the summed connections weights for each connection, default: True
        weight_col (str): name of the column in 'connections' which hold the connection weights, default: "weight"
    
    Returns:
        DataFrame|tuple: a connectivity matrix in form of a data frame with labelled columns and rows, if requested additionally a Series with the 
        total input weights for each neuron in col_grp (before subsetting)
    """
    # Convert the connection data frame to a connectivity matrix, sum weights according to grouping variables
    full_conn_mat = pd.DataFrame(connections.groupby([row_grp,col_grp])[weight_col].sum()).reset_index()
    full_conn_mat = full_conn_mat.pivot(columns=col_grp, index=row_grp, values=weight_col).fillna(0.0)
    # Compute the total number of inputs (summed weights for each column)
    total_input_weights = full_conn_mat.sum(axis=0)

    # Normalize to total number of inputs if requested
    if normalize:
        # Normalize connectivity matrix
        full_conn_mat = full_conn_mat / total_input_weights
    
    if isinstance(subset, list):
        # Create a subset with the neurons of interest
        conn_mat = full_conn_mat.loc[subset[0], subset[1]]
    else:
        conn_mat = full_conn_mat

    # Return the results as requested, with or without the normalization series
    if return_norm_arr:
        return conn_mat, total_input_weights
    else:
        return conn_mat



def flatten(l:list) -> list:
    """### Flatten list of lists

    Args:
        l (list): list to flatten
    Returns:
        flattened list
    """
    flat = []
    # Append non-list elements and extend list with list elements
    for el in l:
        if type(el) is list:
            flat.extend(flatten(el))
        else:
            flat.append(el)
    return flat



def roi_info_summary(roi_infos:list|np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """### Sum up the counts of pre- and postsynaptic sites from a set of roiInfo fields
    The region-of-interest information is assumed to be stored in a dict of dicts like they are presented in neuron information 
    tables retrieved via the *neuprint* interface or via `fetch_all_neuron_props()`. 

    Args:
        roi_infos (list|array): a list or an array containing roiInfo fields from a neuron info table like those retrieved via 
                                the neuprint interface or via `fetch_all_neuron_props()`

    Returns:
        tuple: a tuple of three numpy arrays where the first contains the names of the ROIs, the second the summed number of 
        presynaptic sites and the third the summed number of postsynaptic sites
    
    """
    # Allocate lists to store the results
    rois = []
    all_cpost = []
    all_cpre = []

    # Iterate over the given ROI infos
    for roii in roi_infos:
        # Get the ROIs for the current info field
        curr_rois = roii.keys()

        # Iterate over the ROIs
        for roi in curr_rois:
            # Store the name
            rois.append(roi)

            # Store the number of postsynaptic sites if available, set to 0 otherwise
            if "post" in roii[roi]:
                all_cpost.append(roii[roi]["post"])
            else:
                all_cpost.append(0)

            # Store the number of presynaptic sites if available, set to 0 otherwise
            if "pre" in roii[roi]:
                all_cpre.append(roii[roi]["pre"])
            else:
                all_cpre.append(0)

    # Get the set of unique ROIs
    rois_unique = np.unique(rois)
    # Convert to numpy arrays
    rois = np.array(rois)
    all_cpost = np.array(all_cpost)
    all_cpre = np.array(all_cpre)
    # Sum the counts for pre and postsynaptic sites
    cpost_count = np.array([ np.sum(all_cpost[rois == roi]) for roi in rois_unique ])
    cpre_count = np.array([ np.sum(all_cpre[rois == roi]) for roi in rois_unique ])
    
    return rois_unique, cpre_count, cpost_count
