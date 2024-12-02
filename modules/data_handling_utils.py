""" # Load, extract, copy data 

This module includes the following functions:
    - assemble_metadata:                  create a meta data table for a list of experiments
    - copy_data:                          copy directories 
    - create_tracking_analysis_files:     create yaml analysis files for sleap tracking 
    - extract_tracking_data:              extract the tracking data and store them separately for faster access
    - load_tracking:                      load tracking data saved by the extract_tracking_data() function
    - load_annotations:                   load annotation information and stimulus information
    - load_annotations_simple:            load annotation information in a raw format
    - load_dataset:                       load a dataset (hdf5)
    - load_dataset_xb:                    load a dataset using *xarray_behave*
    - remove_avi_files:                   remove avi video files in directories if a mp4 file is present

Author:         Bjarne Schultze
Last modified:  02.12.2024
"""

import h5py
import pandas as pd
import numpy as np
import shutil
import os
import copy
import xarray_behave as xb
from xarray import Dataset
import modules.analysis_utils as autils


def assemble_metadata(experimenter:str, rig:str, data_path:str, output_path:str="./", schema:str='./schema.csv', extract_info:list[str]=['filename', 'notes', 'experiment group', 
                        'genotype', 'date_eclosed', 'sex', 'housing', 'individuals', 'genotype.1', 'date_eclosed.1', 'sex.1', 'housing.1', 'individuals.1']) -> None:
    """### Assemble the metadata for the experiments in the schema

    Args:
        experimenter (str): name of the experimenter in the schema
        rig (str): name of the setup in the schema
        data_path (str): path to the parent directory for the data directories
        output_path (str): path where the output file is saved
        schema (str): path to the schema
        extract_info (list[str]): list of all features to extract from the schema

    Returns:
        None    
    """
    # Load the schema
    schema_raw = pd.read_csv(schema)
    # Create subset with only my information
    schema_subset = schema_raw.loc[np.logical_and(schema_raw['experimenter'] == experimenter, schema_raw['rig'] == rig)].reset_index()
    # Extract only the relevant information
    exp_metadata = schema_subset[extract_info]

    # Allocate lists to collect information
    stim_ons_all = []
    stim_offs_all = []
    stim_volt_all = []
    sampling_rates = []

    # Print info
    print(f"Extracting stimulus info for {exp_metadata.shape[0]} datafiles. Lean back, this might take a while!")

    # Initialize counters
    counter = 0
    full_counter = 0

    # Iterate over all experiments
    for exp in exp_metadata['filename']:
        # Assemble file path and print current file
        file_path = f'{data_path}{exp}/{exp}_daq.h5'
        print(f'Working on file ({full_counter+1}/{exp_metadata.shape[0]}): {file_path}')

        # Load the current dataset
        with h5py.File(file_path, "r") as f:
            dataset = np.array(f['samples'])    # Sampled data
            sampling_rates.append(round(np.array(f["samplenumber"]).mean()))    # Sampling rate

        # Extract on- and offsets as well as stimulus intensities
        stim_on, stim_off, stim_volt = autils.analyze_light_stim(dataset[:,18])
        # Store all stimulus information
        stim_ons_all.append(stim_on)
        stim_offs_all.append(stim_off)
        stim_volt_all.append(stim_volt)

        # Increment counters
        counter += 1
        full_counter += 1

        # Save temporary data to file for eyery 10 files loaded
        if counter == 5:
            # Make a temporary copy
            exp_metadata_tmp = copy.deepcopy(exp_metadata.loc[:full_counter-1,:])
            # Add the stimulus information to the metadata dataframe
            exp_metadata_tmp['stim_on'] = stim_ons_all
            exp_metadata_tmp['stim_off'] = stim_offs_all
            exp_metadata_tmp['stim_volt'] = stim_volt_all
            # Add the sampling rate
            exp_metadata_tmp['sampling_rate'] = sampling_rates

            # Save temporary metadata dataframe
            exp_metadata_tmp.to_pickle(f"{output_path}/metadata.pkl")
            # Reset counter
            counter = 0

    # Print success message
    print("Successfully collected all stimulus information!")

    # Add the stimulus information to the metadata dataframe
    exp_metadata.loc[:,'stim_on'] = stim_ons_all
    exp_metadata.loc[:,'stim_off'] = stim_offs_all
    exp_metadata.loc[:,'stim_volt'] = stim_volt_all
    # Add the sampling rate
    exp_metadata.loc[:,'sampling_rate'] = sampling_rates

    # Save metadata dataframe
    exp_metadata.to_pickle(f"{output_path}/metadata.pkl")
    # Print finish message
    print(f"Successfully assembled all metadata and saved it to {output_path}/metadata.pkl!")



def copy_data(dirnames:list[str], source_path:str, dest_path:str) -> None:
    """### Copy a list of directories 

    Args:
        dirnames (list[str]): list of names of the directories to copy
        source_path (str): path to the parent source directory
        dest_path (str): path to the parent destination directory
    
    Returns:
        None
    """
    # Set a counter for the copied directories
    dir_counter = 0

    # Iterate over the folder names and copy the experiment folder
    for exp in dirnames:
        source_path_complete = f'{source_path}/{exp}'
        dest_path_complete = f'{dest_path}/{exp}'

        # Copy the directory (override if it exists at destination)
        shutil.copytree(source_path_complete, dest_path_complete, dirs_exist_ok=True)
        # Increase counter
        dir_counter += 1

    # Print success message
    print(f'Successfully copied {dir_counter} directories!')



def create_tracking_analysis_files(filenames:list[str], animal_counts:list|np.ndarray, main_path:str="E:/", dat_path:str="dat/", **kwarg) -> None:
    """### Create analysis profile files for tracking analysis with sleap

    Args:
        filenames (list[str]): list of filenames identifying the experiments to be analyzed
        animal_counts (list, array): gives the number of animals in each experiment listed in "filenames"
        main_path (str): main path where to find the data
        dat_path (str): path where to find the data directory holding the directories in "filenames", relative to "main_path", full path to single 
                        experiments is constructed as f'{main_path}{dat_path}/{filename}/{filename}'
        default_yaml (str, optional): path (full) to a default yaml analysis profile, any {animal_count} instances in this file will be substituted with the 
                                      provided animal counts, if not given a internal default will be used
    
    Returns:
        none, the analysis files will be written to f'{main_path}{dat_path}/{filename}/{filename}_analysis.yaml'
    """
    # Load a default yaml file if a path is given, otherwise use a default template
    if "default_yaml" in kwarg:
        with open(kwarg["default_yaml"]) as f:
            file_content = f.readlines()
    else:
        file_content = ['Animals:\n',
                        '  angles: []\n',
                        '  centers: []\n',
                        '  geometries: []\n',
                        '  nb_rois: 0\n',
                        '  positions: []\n',
                        '  sizes: []\n',
                        'Chambers:\n',
                        '  angles: []\n',
                        '  centers: []\n',
                        '  geometries: []\n',
                        '  nb_rois: 0\n',
                        '  positions: []\n',
                        '  sizes: []\n',
                        'Jobs:\n',
                        '  analyses_profiles: sleapBig.yaml\n',
                        '  sleap:\n',
                        '    batch-size: 8\n',
                        '    max-instances: 16\n',
                        '    modelname: ../snakemake-workflows/sleap/models/sleapBig\n',
                        '    no-empty-frames: true\n',
                        '    tracking.clean_instance_count: {animal_count}\n',
                        '    tracking.match: hungarian\n',
                        "    tracking.pre_cull_iou_threshold: '0.8'\n",
                        '    tracking.pre_cull_to_target: true\n',
                        "    tracking.robust: '0.95'\n",
                        '    tracking.similarity: iou\n',
                        '    tracking.target_instance_count: {animal_count}\n',
                        '    tracking.track_window: 15\n',
                        '    tracking.tracker: simple\n']
    
    # Assemble the destination pathes
    dest_pathes = [ f'{main_path}{dat_path}/{fname}/{fname}_analysis.yaml' for fname in filenames]

    # Iterate over the pathes and save analysis file either for one or two flies
    for dpath, acount in zip(dest_pathes, animal_counts):
        # Adjust the file content by adding the correct animal number
        file_content = [ l.format(animal_count=acount) for l in file_content ]
        # Save the information to a yaml file at the path assambled before
        with open(dpath, "w") as fnew:
            fnew.writelines(file_content)



def extract_tracking_data(dirnames:list[str], main_path="E:/", dat_path="/dat", res_path="/res", pixel_size_mm=(44.92 / 874)) -> None:
    """### Load the datasets, extract the tracking data and save it to separate files
    
    Args:
        dirnames (list[str]): list of experiment names
        main_path (str): root path for the directory where the dat and res directories are stored
        dat_path (str): path for the dat directory (recorded data)
        pixel_size_mm (float): size of a pixel in mm (default: 44.92 mm diameter/874 px dimeter of the chainingmic chamber)
    Returns:
        None
    """
    # Print info
    print(f"Extracting tracking info for {len(dirnames)} experiments. Lean back, this might take a while!")
    # Initialize an empty list
    no_tracks = []

    for i,exp in enumerate(dirnames):
        try:
            # Assemble the dataset
            dataset = xb.assemble(exp, root=main_path, dat_path=dat_path, res_path=res_path, pixel_size_mm=pixel_size_mm)
            # Assemble metrics for the tracking data, set use_true_times to True to get the desired units for the metrics 
            ds_metrics = xb.assemble_metrics(dataset, use_true_times=True)

            # Add pose positions to the metrics dataset
            ds_metrics['positions'] = dataset.pose_positions
            ds_metrics['positions_allo'] = dataset.pose_positions_allo
            ds_metrics['sampletime'] = dataset.sampletime

            # Save the newly combined dataset
            xb.save(f"{main_path}/res/{exp}/{exp}_tracking_data.h5", ds_metrics)
            # Print success message
            print(f"{i+1}/{len(dirnames)} Saved tracking data for experiment {exp}!")
        except:
            # If there is no tracking data save experiment name and print info
            no_tracks.append(exp)
            print(f"{i+1}/{len(dirnames)} Skipped experiment {exp}! There is no tracking data!")

    # Print finish message
    if len(no_tracks) > 0:
        print(f"\nFinished! There were no tracking data for the following experiments: {no_tracks}")
    else:
        print("\nFinished!")



def load_tracking(files:list[str]|str, dataset:str="abs", main_path:str="./", metadata_path:str="./metadata.pkl", 
                  check_full_stim_set:bool=True) -> tuple[list[np.ndarray], list[np.ndarray], list[list]]:
    """### Load a tracking file assembled by the extract_tracking_data function
    This function offers the option the load the metadata file and check if all stimulus intensities were presented at least once before a 
    potential copulation event. No metadata file is required if the stimulus set should not be checked (check_full_stim_set=False). 
    Independent of that, the function subsets the data along the time axis to everything before a potential copulation. Therefore, the 
    annotation file is read from {main_path}/{file}/{file}_annotations.csv (file being an entry of files). 

    Args:
        files (str, list[str]): list of experiment names (without suffix "_annotations.csv")
        dataset (str): name of the dataset to retrieve, options are "abs" for absolute metrics, "rel" for relative metrics, "pos" for positions, 
                       ans "plos_allo" for allocentric positions, default: "abs"
        main_path (str): main path to search for the data directories, files will be loaded from {main_path}/{file}/{file}_tracking_data.h5 
                         (file being an entry of files)
        metadata_path (str): path to the metadata file in pkl format (e.g. created with assemble metadata)
        check_full_stim_set (bool): indicate whether or no to check if all stimulus intensities were presented at least once, default: True

    Returns:
        tuple: with three items
            1. list[array]: tracking data for all experiments, one array per experiment
            2. list[array]: time vectors corresponding to the tracking data
            3. list[list]: lists with indices describing the dimensions of the tracking data arrays
    """
    # Control input type
    if isinstance(files, str):
        files = [ files ]
    elif not isinstance(files, list):
        raise TypeError("Input must be str or list!")
    
    # Load the metadata file if stimulus set should be checked
    if check_full_stim_set:
        metadata = pd.read_pickle(metadata_path)
    else:
        # Empty data frame to avoid 'unbound' errors
        metadata = pd.DataFrame()

    # Allocate lists to collect the results per input file
    tracking_all = []
    time_all = []
    data_indices = []

    # Iterate through all files
    for ifile, file in enumerate(files):
        # Try to load the tracking data
        try:
            tracking_ds = xb.load(f'{main_path}/{file}/{file}_tracking_data.h5', normalize_strings=False)
        # If no file with the given name was found, print a messange and continue
        except FileNotFoundError:
            print(f"There was no tracking data for experiemnt: {file}!")
            continue

        # Read the annotation file
        data_read = pd.read_csv(f"{main_path}/{file}/{file}_annotations.csv")
        # Search for a copulation event
        coup_time = data_read.loc[data_read["name"] == "start_copulation","stop_seconds"].to_numpy()

        # If requested, check if each stimulus intensity was provided at least once before a potential copulation
        if check_full_stim_set:
            # Get the stimulus information
            file_idx = metadata["filename"] == file
            sampling_rate = metadata.loc[file_idx, "sampling_rate"].values[0]
            stim_off_s = metadata.loc[file_idx, "stim_off"].values[0] / sampling_rate
            stim_volt = metadata.loc[file_idx, "stim_volt"].values[0]

            # If there was a copulation in the experiment
            if len(coup_time) > 0:
                # Logical index for stimuli before copulation time
                stim_select = stim_off_s < coup_time[0]
                # Check if each stimulus intensity was presented at least once, if not skip experiment
                if np.unique(stim_volt).shape[0] != np.unique(stim_volt[stim_select]).shape[0]:
                    print(f"Skipped experiment {file}! Incomplete stimulus set before copulation.")
                    continue
                
                # Create a time selection array to restrict the data to before copulation
                time_select = tracking_ds.time < coup_time
            else:
                # In case of no copulation, select everything 
                time_select = tracking_ds.time > -1

        else:
            # If no stimulus checking is requested ...
            # In case of a copulation 
            if len(coup_time) > 0:
                # Create a time selection array to restrict the data to before copulation
                time_select = tracking_ds.time < coup_time
            else:  
                # In case of no copulation, select everything 
                time_select = tracking_ds.time > -1

        # Select the requested data set from the loaded tracking data, restrict along the time axis as defined above
        # For the first file, save the indices of the data
        match dataset:
            case "abs":
                tracking_all.append(tracking_ds.abs_features[time_select, :, :].to_numpy())
                if ifile == 0: data_indices = [tracking_ds.absolute_features.values, ["time", "flies", "absolute_features"]]
            case "rel":
                tracking_all.append(tracking_ds.rel_features[time_select, :, :, :].to_numpy())
                if ifile == 0: data_indices = [tracking_ds.relative_features.values, ["time", "flies", "relative_flies", "relative_features"]]
            case "pos":
                tracking_all.append(tracking_ds.positions[time_select, :, :])
                if ifile == 0: data_indices = [tracking_ds.poseparts.values, tracking_ds.coords.values, ["time", "flies", "poseparts", "coords"]]
            case "pos_allo":
                tracking_all.append(tracking_ds.positions_allo[time_select, :, :].to_numpy())
                if ifile == 0: data_indices = [tracking_ds.poseparts.values, tracking_ds.coords.values, ["time", "flies", "poseparts", "coords"]]

        # Save the time vector for the current file
        time_all.append(tracking_ds.time[time_select])

    return tracking_all, time_all, data_indices



def load_annotations(files:list[str], main_path="./", metadata_path="./metadata.pkl", check_full_stim_set=True) -> tuple[list, list, list, list]|tuple[list, list, list, list, list]:
    """### Load annotation files
    CSV files with annotation information are loaded and combined in a list. If a copulation occurred ("start_copulation"), only stimulus onset and offsets 
    before the copulation are returned.

    Args:
        files (str, list[str]): list of experiment names (without suffix "_annotations.csv")
        main_path (str): path for the parent directory where the annotation files are stored
        metadata_path (str): path for the metadata file

    Returns:
        tuple: with four or five items
            1. list: list of DataFrames with the annotation information [s], one DataFrame per input file
            2. list: list of nd.arrays with stimulus onsets [s], one per input file
            3. list: list of nd.arrays with stimulus offsets [s], one per input file 
            4. list: list of nd.arrays with stimulus voltage [V], one per input file
            5. optional, list: list of filenames for files in which copulation occurred before each stimulus intensity was presented
    """
    # Control input type
    if isinstance(files, str):
        files = [ files ]
    elif not isinstance(files, list):
        raise TypeError("Input must be str or list!")

    # Load the metadata file
    metadata = pd.read_pickle(metadata_path)

    # Allocate lists to collect the results per input file
    annotations = []
    stim_on_all = []
    stim_off_all = []
    stim_volt_all = []
    bad_files = []

    # Iterate through all files
    for file in files:
        # Get the stimulus information
        file_idx = metadata["filename"] == file
        sampling_rate = metadata.loc[file_idx, "sampling_rate"].values[0]
        stim_on_s = metadata.loc[file_idx, "stim_on"].values[0] / sampling_rate
        stim_off_s = metadata.loc[file_idx, "stim_off"].values[0] / sampling_rate
        stim_volt = metadata.loc[file_idx, "stim_volt"].values[0]

        # Read the file
        data_read = pd.read_csv(f"{main_path}/{file}/{file}_annotations.csv") 
        
        # Search for a copulation event
        coup_time = data_read.loc[data_read["name"] == "start_copulation","stop_seconds"].to_numpy()
        # Select the stimulus on- and offsets accordingly 
        if len(coup_time) > 0:
            # Logical index for stimuli before copulation time
            stim_select = stim_off_s < coup_time[0]
            # If requested, skip experiments in which each stimulus intensity was not presented at least once 
            if check_full_stim_set and np.unique(stim_volt).shape[0] != np.unique(stim_volt[stim_select]).shape[0]:
                print(f"Skipped experiment {file}! Incomplete stimulus set before copulation.")
                bad_files.append(file)
                continue
            # Append the stimulus data
            stim_on_all.append(stim_on_s[stim_select])
            stim_off_all.append(stim_off_s[stim_select])
            stim_volt_all.append(stim_volt[stim_select])
            # Append annotations
            annotations.append(data_read)
        else:
            stim_on_all.append(stim_on_s)
            stim_off_all.append(stim_off_s)
            stim_volt_all.append(stim_volt)
            annotations.append(data_read)

    # Also return list of bad files if check_full_stim_set is True
    if check_full_stim_set:
        return annotations, stim_on_all, stim_off_all, stim_volt_all, bad_files
    else:
        return annotations, stim_on_all, stim_off_all, stim_volt_all



def load_annotations_simple(files:list[str], main_path="./") -> list[pd.DataFrame]:
    """### Load annotation files
    CSV files with annotation information are loaded and combined in a list.

    Args:
        files (str, list[str]): list of experiment names (without suffix "_annotations.csv")
        main_path (str): path for the parent directory where the annotation files are stored

    Returns:
        list: list of DataFrames with the annotation information [s], one DataFrame per input file
    """
    # Control input type
    if isinstance(files, str):
        files = [ files ]
    elif not isinstance(files, list):
        raise TypeError("Input must be str or list!")

    # Allocate lists to collect the results per input file
    annotations = [ pd.read_csv(f"{main_path}/{file}/{file}_annotations.csv") for file in files ]

    return annotations



def load_dataset(path:str, filter=False, filter_chans=(0,16), freq_band=(50,1000), quiet=False) -> tuple[np.ndarray, float]:
    """### Load a data file and filter data if requested
    
    Args:
        path (str): path to the data file to load (including suffix)
        filter(bool): states whether to filter the loaded data (default: False)
        filter_chans (tuple/list): range of channels to filter, python indexing (default: (0,16))
        freq_band (tuple/list): range of frequencies in the pass band (default: (50,1000))
        quiet (bool): states whether to print infos (default: False)
    Returns:
        tuple: with two items
            1. array: contains the sampled channels, selected channels are bandpass filtered if requested, time on axis 0, channels on axis 1
            2. float: sampling rate of the recordings
    """
    # Print info if not quiet
    if not quiet:
        print(f'Loading file: {path}')

    # Open data file and extract data
    with h5py.File(path, "r") as f:
        dataset = np.array(f['samples'])
        # Get the sampling rate
        sampling_rate = round(np.array(f["samplenumber"]).mean())

    # Filter data if requested
    if filter:
        # Print info if not quiet
        if not quiet:
            print(f"Filtering channels {filter_chans[0]} to {filter_chans[1]-1}")
        dataset[:, filter_chans[0]:filter_chans[1]] = autils.filter_song(dataset[:, filter_chans[0]:filter_chans[1]], freq_low=freq_band[0], freq_high=freq_band[1], fs=sampling_rate)

    return dataset, sampling_rate



def load_dataset_xb(expname:str, main_path:str="E:/", dat_path:str="/dat", res_path:str="/res", 
                    include_tracking:bool=True, pixel_size_mm:float|None=(44.92 / 874)) -> tuple[Dataset, Dataset] | Dataset:
    """### Load a dataset and assemble the tracking metrics

    Args: 
        expname (str): name of the experiment directory
        main_path (str): path to the directory hosting the data and results directories
        dat_path (str): relative path to the data directory (relative to main path)
        res_path (str): relative path to the results directory (relative to main path)
        include_tracking (bool): indicates whether to load and return the tracking data (default: True)
        pixel_size_mm (float): size of one pixel in mm, default: 44.9s mm diameter of the chainingmic chamber/874 pixel diameter in videos
    Returns:
        tuple: one/two xarray dataset/s, the first containing the recorded audio traces and several additional information,
               the second containing the tracking information and computed metrics (only if include_tracking=True)
    """
    # Assemble the dataset
    dataset = xb.assemble(expname, root=main_path, dat_path=dat_path, res_path=res_path, pixel_size_mm=pixel_size_mm)
    # Load the tracking data if requested
    if include_tracking:
        # Assemble metrics for the tracking data
        ds_metrics = xb.assemble_metrics(dataset, use_true_times=True)
        # Return data
        return dataset, ds_metrics
    else:
        # Return data
        return dataset



def export_to_hdf5(filepath:str, data:np.ndarray|list|tuple, mode:str="w", **kwarg) -> None:
    """### Export a dataset to a HDF5 file

    Args:
        filepath (str): the path were to store/from where to open the file
        data: the data to store in the file, possible types are np.ndarray, list, tuple, float, int, str
        mode (str): the mode to open the file, possible options: "w" - open to write, overwrite if exists, 
                    "a" - open to write, append if exists
        label (str, optional): specifying the name and location of the dataset within the file, defaults to "/dataset"
        dim_labels (list[str], optional): labels for the dimensions of the dataset, should match the number of dimensions
        attributes (dict, optional): attributes to associate with the dataset, dict keys are used as attribute names, 
                                     dict values are used as the attribute values

    Returns:
        None
    """  
    # Set a name and location for the dataset
    if "label" in kwarg:
        label = f"/{kwarg["label"]}"
    else:
        label = "/dataset"

    try:
        # Open a data file in specified mode
        with h5py.File(filepath, mode) as fdat:

            # Store the counts of all possible changes
            fdataset = fdat.create_dataset(label, data=data)

            # Label the dimensions if labels are given
            if "dim_labels" in kwarg:
                for idim in range(fdataset.ndim):
                    fdataset.dims[idim].label = kwarg["dim_labels"][idim]
            
            # Store attributes if requested
            if "attributes" in kwarg and isinstance(kwarg["attributes"], dict):
                for key, val in kwarg["attributes"].items():
                    fdataset.attrs.create(key, val)

    # Catch IO error and other errors, print a warning that the data is not saved
    except IOError as e:
        print("Caution, data not saved!", e)
    except KeyError as e:
        print("Caution, data not save! Pay attention to use the right keywords.", e)
    except Exception as e:
        print("Caution, data not saved!", e)



def read_hdf_to_dict(filepath:str, include_attr:bool=True) -> dict:
    """### Read a HDF file and store all datasets in a dictionary

    Args:
        filepath (str): path to the HDF5 file to open
        include_attr (bool): indicate whether to store the contentes of the attributes as well,
                             default: True

    Returns:
        dict: dictionary with the dataset locations (names) as keys and the data as values.    
    """

    def save_to_dict(name)-> None:
        """Save the dataset with the given name to the data dictionary"""
        # Check if object is a dataset, if so store it in the dictionary
        if isinstance(fdat[name], h5py.Dataset):
            datadict[name] = np.array(fdat[name])

            if include_attr:
                # Store the attributes as well if requested
                for attr in fdat[name].attrs.keys():
                    datadict[f"{name}_{attr}"] = fdat[name].attrs[attr]

    # Initialize a dictionary to collect the data
    datadict = {}

    try:
        # Load the file
        with h5py.File(filepath, "r") as fdat:
            # Visit all objects in the file
            fdat.visit(save_to_dict)

    # Catch IO and other errors
    except IOError as e:
        print("File not readable!", e)
    except Exception as e:
        print("An error occurred", e)

    return datadict



def remove_avi_files(dat_dir:str, safe:bool=True, check_for:str="mp4") -> None:
    """### Remove avi video files from data directories

    Args:
        dat_dir (str): path to the directory that holds the single data directories
        safe (bool): state whether to check for the existence of a file with the extension *check_for* before removing the avi file, 
                     avi file is only removed if another file with the given extension is found, this is silent, default: True
        check_for (str): the extension to check for before removing the avi file

    Returns:
        None    
    """
    # Iterate over the data dictionaries
    for ddir in os.listdir(dat_dir):
        # If safe mode is selected, check if other format is present before deleting
        if safe:
            if os.path.exists(f"{dat_dir}/{ddir}/{ddir}.avi") and os.path.exists(f"{dat_dir}/{ddir}/{ddir}.{check_for}"):
                # Remove the avi file
                os.remove(f"{dat_dir}/{ddir}/{ddir}.avi")
        else:
            # Remove the avi file
            os.remove(f"{dat_dir}/{ddir}/{ddir}.avi")