"""# Analyze annotated behavioral data
This module contains functions to analyze behavioral datasets. It mainly relies on annotations of events and focusses 
on analyzing the amount and transitions of the annotated events. 

Content:
    - analyze_light_stim:      extract on- and offsets of a stimulus
    - filter_song:             bandpass filter an audio trace
    - cut_events:              cut events out of an recorded trace
    - psth_simple:             compute a Peri-stimulus-time histogram (PSTH) from annotations   
    - psth:                    compute a PSTH of annotations from multiple experiments
    - events_per_stim:         count the events per stimulus presentation
    - segment_per_stim:        get the length of signal segments during stimulus presentations
    - slice_at_stim:           slice a trace (over time) around stimuli 
    - annotation_fi_curve:     calculate an stimulus-response curve from annotations
    - find_event_trains:       find trains of events which are separated by a maximal distance
    - find_event_trains_multi: find trains of events from a raster matrix with multiple trials
    - overlap:                 determine the overlap between two segments
    - raster_mat_from_trains:  create a raster matrix from event train start and stop values
    - multi_trial_iei:         calculate inter-event-intervals from a raster matrix with multiple trials
    - get_signal_switches:     find changes between signal types from a raster matrix with multiple trial
    - count_signal_switches_exp_wise:  count changes between signal types per experiment from a ordered raster matrix
    - contdata_fi_curve:       calculate a stimulus-response curve from continuous data

Author:         Bjarne Schultze
Last modified:  02.12.2024
"""


import numpy as np
import pandas as pd
from itertools import accumulate, groupby
import scipy



def analyze_light_stim(light_chan:np.ndarray, threshold:float=0.04) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """### Analyze the light stimulus to get on- and offsets as well as stimulation intensities

    Args:
        light_chan (n*1 array): channel with the stimulus controlling the light, n data points

    Returns:
        tuple: with three items
            - (m items): indices of all m stimulus onsets
            - (m items): indices of all m stimulus offsets
            - (m items): stimulation voltage for all m stimuli (same order as onsets and offsets)
    """
    # Identify time points where stimulation was applied
    stim_ind_log = light_chan > 0 + threshold
    # Get on- and offsets for all stimulus pulses
    stim_onoff = [0] + list(accumulate(sum(1 for _ in g) for _,g in groupby(stim_ind_log)))
    # Divide into onsets and offsets
    stim_on = [ i for i in stim_onoff[:-1] if stim_ind_log[i] ]
    stim_off = [ i for i in stim_onoff[1:-1] if not stim_ind_log[i] ]

    # Add the last index as the last stimulus offset if the recording ended before the stimulus
    if len(stim_on) > len(stim_off):
        stim_off.append(np.shape(light_chan)[0]-1)

    # Convert to arrays
    stim_on = np.array(stim_on) 
    stim_off = np.array(stim_off)

    # Sort out all artefacts by selecting only stimuli with distances greater than 10 data points
    good_inter_stim = stim_on[1:] - stim_off[:-1] > 10
    stim_on = np.append(stim_on[0],stim_on[1:][good_inter_stim])
    stim_off = np.append(stim_off[:-1][good_inter_stim],stim_off[-1])

    # Sort out remaining artefacts by selecting only stimuli with lengths greater than 10 data points
    good_stim_len = stim_off - stim_on > 10
    stim_on = stim_on[good_stim_len]
    stim_off = stim_off[good_stim_len]

    # Get all stimulation values (in order of appearance) [V]
    stim_volt = np.array([ np.mean(light_chan[i:j]).round(2) if max(light_chan[i:j]) >= 0.2 else np.mean(light_chan[i:j]).round(3) for i,j in zip(stim_on, stim_off) ])
    
    return stim_on, stim_off, stim_volt



def filter_song(data_in:np.ndarray, freq_low:float=50, freq_high:float=1000, fs:int=10000) -> np.ndarray:
    """### Bandpass filter data
    A non-causal (forward + backward filtering) Butterworth bandpass filter is applied to the data. The data is automatically mean-centered.

    Args:
        data_in (n*m array): input data of length n in m different channels
        freq_low (float): lower cutoff frequency, in Hz (default: 50)
        freq_high (float): upper cutoff frequency, in Hz (default: 1000)
        fs (float): sampling rate, in Hz (default: 10000)

    Returns:
        array (n*m): filtered data, same shape as 'data_in'
    """
    # Design a Butterworth bandpass filter
    bp_filter = scipy.signal.butter(5, [freq_low,freq_high], 'bandpass', output='sos', fs=fs)
    # Filter input data (along axis 0) in non-causal way
    filtered_data = scipy.signal.sosfiltfilt(bp_filter, data_in, axis=0)

    return filtered_data



def cut_events(data_in:np.ndarray, event_times:np.ndarray|list, cut_len:float=15, fs:int=10000, norm:bool=False) -> tuple[np.ndarray, np.ndarray]:
    """### Cut events from an input signal and create sections with the event centered
    
    Args:
        data_in (n*m array): data trace with the events
        event_times (k*1 array|list): times of the k events [s]
        cut_len (float/int): section to cut around the event, in ms, section will be 2*cut_len
        fs (float/int): sampling rate, in Hz (default: 10000)
        
    Returns:
        tuple: with two arrays
            - (k*l): cut sections of the k events with the peak centered and positive, sections of l data points
            - (l*1): time vector corresponding to the events with peak at 0, l time points [ms]
    """
    # Convert event times to data points, round to ensure valid indices
    event_indices = [ int(round(evtime*fs)) for evtime in event_times ]
    # Convert cut_len from ms to data points
    cut_len = int(round(cut_len / 1000 * fs))

    # Find the loudest channel for each event
    max_chans = [ np.argmax(np.max(np.abs(data_in[event-cut_len:event+cut_len,:]), axis=0)) for event in event_indices ]

    # Find the exact index for the maximum of each event
    event_indices_exact = [ np.argmax(np.abs(data_in[event-cut_len:event+cut_len,max_chan]), axis=0) + (event-cut_len) for event,max_chan in zip(event_indices,max_chans) ]
    # Cut the input signal according to the exact event indices and the defined cut length (section length 2*cut_len)
    cutouts_raw = [ data_in[event-cut_len:event+cut_len,chan] for event,chan in zip(event_indices_exact,max_chans) ]
    # Invert the section if needed to ensure positive peaks
    cutouts_pos = [ -cutout if np.abs(np.min(cutout)) > np.abs(np.max(cutout)) else cutout for cutout in cutouts_raw ]

    # Normalize the cutouts to their maximal value
    if norm:
        cutouts = np.array([ cutout / np.max(cutout) for cutout in cutouts_pos ])       # What about the Euclician norm?
    else:
        cutouts = np.array(cutouts_pos)

    # Create suitable time vector [ms]
    cutout_time = np.arange(-cutouts.shape[1]/2, cutouts.shape[1]/2, 1) / fs *1000
        
    return cutouts, cutout_time



def psth_simple(events:np.ndarray, stim_onsets:np.ndarray, stim_len:float, bin_width:float=0.05, 
                padding_pre:float=5.0, padding_post:float=5.0, output:str='average') -> tuple[np.ndarray, np.ndarray]: 
    """### Compute a peri-stimulus-time histogram

    Args:   
        events (n*1 array): times of n events [s]
        stim_onsets (m*1 array): times of m stimulus onsets [s]
        stim_len (float): length of stimuli [s]
        bin_width (float): width of each bin [s] (default: 0.05 s)
        padding_pre (float): time before the stimulus to be included into the PSTH [s] (default: 5 s)
        padding_post (float): time after the stimulus to be included into the PSTH [s] (default: 5 s)
        output (str): average, sum, norm, full

    Returns:
        tuple: with two arrays
            - (k*1): histogram, bin count based on the event times, k depends on stim_len and bin_width
            - (k+1)*1: edges for the histogram, in s, includes the rightmost bin hence k+1 entries
    """
    # Allocate list for storage
    events_binned = []
    # Define the bin edges for the histogram computation
    bin_edges = np.arange(-padding_pre, stim_len+padding_post+bin_width, bin_width)

    # Iterate over all stimuli
    for stim in stim_onsets:
        # Create a subset of events during stimulation including a padding before and after stimulation, subtract stimulus onset
        trial_events = events[np.logical_and(events > stim-padding_pre, events < stim+stim_len+padding_post)] - stim
        # Compute the histogram and collect the counts for the bins
        binned, _ = np.histogram(trial_events, bins=bin_edges)
        events_binned.append(binned)
    # Convert to numpy array
    events_binned = np.array(events_binned)

    # Calculate the requested ouput format
    if output == 'sum':        # Counts summed within each bin over all trials
        hist = np.sum(events_binned, axis=0)
    elif output == 'average':  # Counts averaged within each bin over all trials
        hist = np.mean(events_binned, axis=0)
    elif output == 'norm':     # Noramlized to the fraction of trials in which events occured 
        # Flatten the binned event times
        events_b_flat = events_binned.reshape(-1)
        # Convert counts to binary array
        events_binary = np.array([ 1 if x > 0 else 0 for x in events_b_flat ])
        # Re-shape to recover the trial*bin_count shape 
        events_binary = events_binary.reshape(events_binned.shape)
        # Average over all trials to get the fraction of trials with events
        hist = np.mean(events_binary, axis=0)
    else:
        hist = events_binned  # Retain the results of the single trials

    return hist, bin_edges



def psth(anno_tables:list[pd.DataFrame], stim_onsets:list[np.ndarray], stim_len:float, event_label:str, bin_width:float=0.05, 
             padding_pre:float=5.0, padding_post:float=5.0, output:str='average') -> tuple[np.ndarray, np.ndarray]: 
    """### Compute a peri-stimulus-time histogram for multiple experiments at once
    
    Args:   
        anno_tables (list[DataFrame]): list of annotation data frames, one for each experiment
        stim_onsets (list[array]): list of arrays containing the stimulus onsets for each experiment
        stim_len (float): length of stimuli [s]
        event_labels (str): name of the annotation label to be analyzed
        bin_width (float): width of each bin [s] (default: 0.05 s)
        padding_pre (float): time before the stimulus to be included into the PSTH [s] (default: 5 s)
        padding_post (float): time after the stimulus to be included into the PSTH [s] (default: 5 s)
        output (str): average, sum, norm, full

    Returns:
        tuple: with two arrays
            - (k*1): histogram, bin count based on the event times, k depends on stim_len and bin_width
            - ((k+1)*1): edges for the histogram, in s, includes the rightmost bin hence k+1 entries
    """
    # Allocate list for storage
    events_binned = []
    # Define the bin edges for the PSTH
    bin_edges = np.arange(-padding_pre, stim_len+padding_post+bin_width, bin_width)

    for events, stim_on in zip(anno_tables, stim_onsets):
        # If the event-label contains 'sine', create event-like array for the sine segments
        if 'sine' in event_label:
            # Get start and stop seconds
            sine_start = events.loc[events['name'] == event_label, 'start_seconds'].to_numpy()
            sine_stop = events.loc[events['name'] == event_label, 'stop_seconds'].to_numpy()
            
            # Create an array with time points filling the time difference between sine start and stop (based on the defined bin width)
            if sine_start.size == 0:
                event_times = sine_start
            else:
                event_times = np.concatenate([ np.arange(start,stop,bin_width) for start,stop in zip(sine_start,sine_stop) ])
        else:
            # Get the start times for all other events
            event_times = events.loc[events['name'] == event_label, 'start_seconds'].to_numpy()

        # Calculate the PSTH
        binned, _ = psth_simple(event_times, stim_on, stim_len, bin_width=bin_width, padding_pre=padding_pre, 
                                 padding_post=padding_post, output="full")
        # Store the results
        events_binned.append(binned)
    
    # Combine the single experiment PSTHs to one matrix if non-empty
    if len(events_binned) > 0: 
        events_binned = np.vstack(events_binned) 
    else: 
        events_binned = np.array(events_binned)

    # Calculate the requested ouput format
    if output == 'sum':        # Counts summed within each bin over all trials
        hist = np.sum(events_binned, axis=0)
    elif output == 'average':  # Counts averaged within each bin over all trials
        hist = np.mean(events_binned, axis=0)
    elif output == 'norm':     # Noramlized to the fraction of trials in which events occured 
        # Convert counts to binary array
        events_binary = (events_binned > 0).astype('int')
        # Average over all trials to get the fraction of trials with events
        hist = np.mean(events_binary, axis=0)
    else:
        hist = events_binned  # Retain the results of the single trials

    return hist, bin_edges



def events_per_stim(event_times:np.ndarray, stim_on:np.ndarray, stim_off:np.ndarray, stim_volt:np.ndarray, relative:bool=True) -> np.ndarray:
    """### Calculate the average event count per stimulus 

    Args:
        event_times (n*1 array): times of n events [s]
        stim_on (m*1 array): times of m stimulus onsets [s]
        stim_off (m*1 array): times of m stimulus offsets [s]
        stim_volt (m*1 array): voltage for each of the m stimuli [V]
        relative (bool): indicates whether to calculate counts relative to time before stimulus (default: True)

    Returns:
        array: average event counts during stimuli 
    """
    # Count the events during stimulation 
    event_counts = np.array([ sum(np.logical_and(event_times >= on, event_times <= off)) for on, off in zip(stim_on, stim_off) ])

    # Relativize if requested
    if relative:
        # Count the events preceding the stimulation
        event_counts_pre = np.array([ sum(np.logical_and(event_times < on, event_times >= on-(off-on))) for on, off in zip(stim_on, stim_off) ])
        # Subtract the number of events preceding the stimuli from the number during stimulation
        event_counts = event_counts - event_counts_pre

    # Average across all stimuli of the same intensity
    mean_event_counts = np.array([ np.mean(event_counts[stim_volt == v]) for v in np.unique(stim_volt) ])

    return mean_event_counts



def segment_per_stim(seg_starts:np.ndarray, seg_stops:np.ndarray, stim_on:np.ndarray, stim_off:np.ndarray, stim_volt:np.ndarray, relative:bool=True, 
                     sampling_rate:int=10000, exp_len:float=1799) -> np.ndarray:
    """### Calculate the duration of a segment during a stimulus (or its increase to before stimulation)

    Args: 
        seg_starts (n*1 array): start times of n segments [s]
        seg_stops (n*1 array): stop times of n segments [s]
        stim_on (m*1 array): times of m stimulus onsets [s]
        stim_off (m*1 array): times of m stimulus offsets [s]
        stim_volt (m*1 array): voltage for each of the m stimuli [V]
        relative (bool): indicates whether to calculate durations relative to time before stimulus (default: True)
        sampling_rate (int): sampling rate of recording [Hz]
        exp_len (float): length of the recording [s]

    Returns:
        array: average duration of sine during stimuli [s]
    """

    # Create zero vector of the recording length with one entry per sample
    seg_binary = np.zeros(exp_len*sampling_rate)

    # Create a binary vector with a 1 whenever there was sine at a sample point
    for start, stop in zip(seg_starts, seg_stops):
        seg_binary[round(start*sampling_rate):round(stop*sampling_rate)] = 1

    seg_len = []
    # Collect the segment seconds per stimulus 
    for on,off in zip(stim_on*sampling_rate,stim_off*sampling_rate):
        seg_len.append(np.sum(seg_binary[round(on):round(off)]) / sampling_rate)
    # Convert to array
    seg_len = np.array(seg_len)

    # Relativize the segment length if requested
    if relative:
        seg_len_pre = []
        # Collect the segment seconds per stimulus
        for on,off in zip(stim_on*sampling_rate,stim_off*sampling_rate):
            seg_len_pre.append(np.sum(seg_binary[round(on-(off-on)):round(on)]) / sampling_rate)

        # Convert to array
        seg_len_pre = np.array(seg_len_pre)
        # Subtract count before stimuli from count during stimuli
        seg_len = seg_len - seg_len_pre

    # Average the segment length over all stimuli of the same intensity
    mean_seg_len = np.array([ np.mean(seg_len[stim_volt == v]) for v in np.unique(stim_volt) ])

    return mean_seg_len



def slice_at_stim(data:np.ndarray, stim_onsets:np.ndarray, stim_offsets:np.ndarray, padding_pre:float=5, padding_post:float=5, 
                  output:str="average", sampling_rate:int=10000) -> np.ndarray: 
    """### Slice an array around a set of stimuli
    Ignores nan values for averaging.
    
    Args:   
        data (n*1 array): array with an arbitrary measurement (e.g. velocity)
        stim_onsets (m*1 array): times of m stimulus onsets [s]
        stim_offsets (m*1 array): times of m stimulus onsets [s]
        padding_pre (float/int): time before the stimulus to be included into the PSTH [s] (default: 5 s)
        padding_post (float/int): time after the stimulus to be included into the PSTH [s] (default: 5 s)
        output (str): average, full

    Returns:
        array: sliced input data, averaged over axis 0 if output="average", otherwise the array has the m trials along axis 0
    """
    # Convert stimulus on- and offset times to data points
    stim_on = np.round((stim_onsets - padding_pre) * sampling_rate).astype('int')
    stim_off = np.round((stim_offsets + padding_post) * sampling_rate).astype('int')

    # Collect the slices of the data
    slices = [ data[on:off] for on, off in zip(stim_on, stim_off) ]

    # Ensure same length before stacking the slices into one array
    slice_len = stim_off[0]-stim_on[0]
    slices_ok = [ s for s in slices if s.shape[0] == slice_len ]
    # Convert to numpy array
    slices_ok = np.array(slices_ok)

    # Calculate the requested output format
    if output == "average":
        data_out = np.nanmean(slices_ok, axis=0)
    else:
        data_out = slices_ok  # Retain the results of the single trials

    return data_out



def annotation_fi_curve(annotations:list[pd.DataFrame], stim_ons:list[np.ndarray], stim_offs:list[np.ndarray], stim_volts:list[np.ndarray], 
                       relative:bool=True) -> tuple[list[np.ndarray],list[np.ndarray],list[np.ndarray]]:
    """### Quantify song and vibrations for different stimulus intensities and experiments
    This function calculates the data for a FI curve which shows the amount of pulse/sine/vibrations for 
    a given set of stimulus intensities. 

    Args:
        annotations (list[pd.DataFrame]): list of data frames containing the annotations, each data frame representing one experiment
        stim_ons (list[array]): list of arrays containing the stimulus onsets for each experiment [s]
        stim_offs (list[array]): list of arrays containing the stimulus offsets for each experiment [s]
        stim_volts (list[array]): list of arrays containing the stimulus intensities for each experiment [V]
    
    Returns:
        tuple: with three items
            - list of pulse counts grouped by stimulus intensity [pulses/min]
            - list of vibration counts grouped by stimulus intensity [vibrations/min]
            - list of sine amounts grouped by stimulus intensity [seconds sine/min]
    """
    # Initialize lists to collect the song counts
    p_counts = []
    vib_counts = []
    sine_counts = []

    # Calculate a multiplier to convert counts/duration per stimulus to counts/duration per minute
    min_multiplier = 1/(stim_offs[0][0]-stim_ons[0][0]) * 60

    # Iterate over the annotations per experiment
    for ann, st_on, st_off, st_volt in zip(annotations, stim_ons, stim_offs, stim_volts):
        # Extract pulse times and calculate pulses per stimulus 
        p_times = ann.loc[ann["name"] == "pulse_manual", "start_seconds"].to_numpy()
        p_counts.append(events_per_stim(p_times, st_on, st_off, st_volt, relative=relative) * min_multiplier)
        # Repeat for vibrations
        vib_times = ann.loc[ann["name"] == "vibration_manual", "start_seconds"].to_numpy()
        vib_counts.append(events_per_stim(vib_times, st_on, st_off, st_volt, relative=relative) * min_multiplier)
        # Extract and quantify amout of sine per stimulus
        sine_index = ann["name"] == "sine_manual"
        sine_starts = ann.loc[sine_index, "start_seconds"].to_numpy()
        sine_stops = ann.loc[sine_index, "stop_seconds"].to_numpy()
        sine_counts.append(segment_per_stim(sine_starts, sine_stops, st_on, st_off, st_volt, relative=relative) * min_multiplier)

    # Stack the results for the single experiments
    p_counts_c = np.hstack(p_counts)
    vib_counts_c = np.hstack(vib_counts)
    sine_counts_c = np.hstack(sine_counts)
    stim_volt_c = np.hstack([np.unique(x) for x in stim_volts])

    # Group the results per stimulus intensity
    p_counts_grp = [ p_counts_c[stim_volt_c == v] for v in np.unique(stim_volt_c) ]
    vib_counts_grp = [ vib_counts_c[stim_volt_c == v] for v in np.unique(stim_volt_c) ]
    sine_counts_grp = [ sine_counts_c[stim_volt_c == v] for v in np.unique(stim_volt_c) ]

    return p_counts_grp, vib_counts_grp, sine_counts_grp



def find_event_trains(event_times:np.ndarray, cutoff:float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """### Find trains of distinct single events separated by a maximal distance

    Args:
        event_times (array): the times of the single events (time points) [s]
        cutoff (float): the maximally allowed distance (time difference) between two adjacent events [ms]

    Returns:
        tuple: with tree items
            - array (n*1): start times of all n event trains
            - array (n*1): end times of all n event trains
            - array (n*1): the number of single events per train for all n trains    
    """
    # Calculate the inter-event-intervals
    ieinterval = np.diff(event_times) * 1000
    # Allocate lists to gather results
    train_start = []
    train_stop = []
    train_evt_count = []
    # Set a counter and a helper variable
    new_train= True
    event_counter = 0

    # Iterate over the event times
    for ei,evt in enumerate(event_times[:-1]):
        # Get the interval to the nex event
        interval = ieinterval[ei]

        # Start a new train if interval below cutoff
        if new_train and interval <= cutoff:
            train_start.append(evt)
            # Set decision variable to False and increase event counter
            new_train = False
            event_counter +=1
        # Count event if interval below cutoff
        elif interval <= cutoff:
            event_counter += 1
        # If a train was started and the interval is above the cutoff, end the train
        elif not new_train and interval > cutoff:
            train_stop.append(evt) 
            # Reset the decision variable and increase the event counter
            new_train = True
            event_counter += 1
            # Save the train length and reset the event counter
            train_evt_count.append(event_counter)
            event_counter = 0

        # For the last event, check if it is part of the previous train 
        if not new_train and ei+1 == event_times.shape[0]-1:
            # If so, end the train, increase the counter, and save the train length
            train_stop.append(event_times[ei+1])
            event_counter += 1
            train_evt_count.append(event_counter)

    # Convert lists to numpy arrays and return
    return np.array(train_start), np.array(train_stop), np.array(train_evt_count)



def find_evt_trains_multi(event_mat:np.ndarray, time_vec:np.ndarray, cutoff:float)-> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]: 
    """### Find event trains from a matrix with multiple trials

    Args:
        event_mat (array, n*m): matrix containing raster-plot-like information for n trails of length m
        time_vec (array, m*1): time vector corresponding to axis 1 of the event_mat
        cutoff (float): inter-event-interval at which a event train is considered to end [ms]

    Returns:
        tuple: with three items
            - list (len n) with start times of the trains (multiple possible per trial)
            - list (len n) with end times of the trains 
            - list (len n) with train lengths measured as event counts    
    """
    # Check for a suitable time vector
    if event_mat.shape[1] != time_vec.shape[0]:
        raise ValueError(f"The time vector must match the event matrix along axis 1 (shape:{event_mat.shape})!")
    
    # Allocate lists to collect the results
    train_start = []
    train_stop = []
    train_len = []

    # Iterate over the trials (rows of the full hist matrix)
    for i in range(event_mat.shape[0]):
        # Get the pulse, vibration and sine times
        evt_times = time_vec[(event_mat > 0)[i,:]]

        # Find trains of each signal type
        t_start, t_stop, t_len = find_event_trains(evt_times, cutoff)

        # Append the start, end, and length values for the signal trains
        train_start.append(t_start)
        train_stop.append(t_stop)
        train_len.append(t_len)

    return train_start, train_stop, train_len



def overlap(start1:list|np.ndarray, stop1:list|np.ndarray, start2:list|np.ndarray, stop2:list|np.ndarray) -> np.ndarray:
    """### Determine the overlap between two segments

    Args:
        start1 (list/array): start values of type 1 segments
        stop1 (list/array): stop value of type 1 segments
        start2 (list/array): start values of type 2 segments
        stop2 (list/array): stop values of type 2 segments
    
    Return:
        array: start and stop values for the overlap     
    """
    # Allocate list to store results
    overlapping = []
    # Iterate over the first start-stop pairs
    for on1,off1 in zip(start1,stop1):
        # Iterate over the second start-stop pairs
        for on2,off2 in zip(start2,stop2):
            # Check for an overlap of the two event trains
            if on1 >= on2 and off1 <= off2:                     # case: 2 1-1 2
                overlapping.append([on1,off1])
            elif on1 >= on2 and off2 > on1 and off1 >= off2:    # case: 2 1-2 1
                overlapping.append([on1,off2])
            elif on2 >= on1 and off2 <= off1:                   # case: 1 2-2 1
                overlapping.append([on2,off2])
            elif on2 >= on1 and off1 > on2 and off2 >= off1:    # case: 1 2-1 2
                overlapping.append([on2,off1])
    
    return np.array(overlapping)



def raster_mat_from_trains(train_starts:np.ndarray|list, train_stops:np.ndarray|list, time_vec:np.ndarray, shape:tuple[int,int], fill:int=1) -> np.ndarray:
    """### Create a raster matrix from event trains (start, stop values)

    Args:
        train_starts (list, len n): list of train start values (as list of array) for each of the n trials
        train_stops (list, len n): list of train stop values (as list of array) for each of the n trials
        time_vec (array): time vector corresponding to dimension 1 of the raster matrix
        shape (tuple): shape of the matrix that will be created, numpy notation
        fill (int): a number to place at each time point where a event train was found

    Returns:
        array: raster matrix with 'fill' values at all positions with events, shape is 'shape'    
    """
    # Create an zero-matrix
    raster_mat = np.zeros(shape)

    # Iterate over all trials (rows)
    for triali in range(raster_mat.shape[0]):
        # Place the trains in the matrix
        for train_start, train_stop in zip(train_starts[triali], train_stops[triali]):
            raster_mat[triali, np.logical_and(time_vec >= train_start, time_vec <= train_stop)] = fill
    
    return raster_mat



def multi_trial_iei(time_vec:np.ndarray, raster_mat:np.ndarray, cutoff:float, abs_iei:bool=True) -> np.ndarray:
    """### Calculate inter-event-intervals for multiple trials 

    Args:
        time_vec (m*1 array): time vector corresponding to axis 1 (m) of the raster matrix [s]
        raster_mat (n*m array): zero-non-zero matrix indicating the times of events in n trials over m sampling points
        cutoff (float): maximal inter-event-interval to be considered
        abs_iei (bool): indicate whether or not to use the absolute value for the inter-event-intervals

    Returns:
        array: inter-event-intervals over all trials [ms]    
    """
    # Get the event times from the raster matrix (trial/row-wise)
    event_times = [ time_vec[(raster_mat > 0)[i,:]] for i in range(raster_mat.shape[0]) ]

    # Calculate the inter-event-intervals (trial-wise), make intervals absolute, if requested
    if abs_iei:
        ieinterval = [ np.abs(np.diff(evts)) * 1000 for evts in event_times ]   # [ms]
    else:
        ieinterval = [ np.diff(evts) * 1000 for evts in event_times ]   # [ms]
    
    # Sort intervals according to to cutoff value
    ieinterval_srt = np.hstack([ iei[iei <= cutoff] for iei in ieinterval ])
    
    return ieinterval_srt



def get_signal_switches(raster_mats:list[np.ndarray], bin_width:float, tolerance:float=0.1, 
                        overlap_conversion:dict={4.0:np.array([1.0,3.0]), 5.0:np.array([3.0,2.0])}) -> tuple[np.ndarray, np.ndarray]:
    """### Find all changes of signal type from raster matrices
    For each signal type a zero-non-zero matrix must be given which is non-zero during signal trains. Each matrix should be filled with 
    a different number. The recommendation would be pulse as 1, sine as 2 and vibrations as 3. Changes are only saved is the signal type 
    changes. If there is an overlap, this is split apart and two changes are stored. In such cases it might therefor happen that changes 
    with equal pre- and post-change type are stored, but this only happens for signal type overlaps.
    
    Args:
        raster_mats (list[n*m array]): list of zero-non-zero matrices indicating the times of events in n trials over m sampling points
        bin_width (float): bin width that was used to create the raster matrices [s]
        tolerance (float): time between two signals that should not be considered [s], default: 0.1 s
        overlap_conversion (dict): map an overlap type to the two types it consists of

    Returns:
        tuple: with two items
            - array (n*2): n signal switches, organized with rows being changes and columns [type_prior_change, type_post_change]
            - array (n*2): n indices for the n signal switches, rows are changes, first column is the row, second column the column index
    """
    # Sum the raster matrices along axis 0 (conserves input shape n*m)
    raster_mat = np.sum(raster_mats, axis=0)
    # Calculate the difference along axis 1 (within each trial)
    raster_mat_diff = np.diff(raster_mat, axis=1)
    
    # Get the indices for the signal changes
    change_idx = np.argwhere(raster_mat_diff)
    # Split the indices in row and column indices
    idx_r = change_idx[:, 0] 
    idx_c = change_idx[:, 1]

    # Convert the given tolerance value in data points
    tol_dp = tolerance / bin_width
    
    # Allocate a list to store the signal type changes
    changes = []
    indices = []

    # Loop until the index vector is empty
    while idx_r.shape[0] > 0:
        # Get the row and column index of the current change (always at position 0)
        idx_curr_r = idx_r[0]
        idx_curr_c = idx_c[0]

        # Get the type prior to and after the change
        changefrom = raster_mat[idx_curr_r, idx_curr_c]
        changeto = raster_mat[idx_curr_r, idx_curr_c+1]

        # If the signal switches to zero, check for other switches in the tolerance window
        if changeto == 0.0:
            # Get all change indices within the current trial
            all_trial_changes = idx_c[idx_r == idx_curr_r]
            # Search for a change in the tolerance window
            next_change = np.where(np.logical_and(all_trial_changes <= idx_curr_c + 1 + tol_dp, 
                                                all_trial_changes > idx_curr_c + 1))[0]
            # If there was a change ...
            if next_change.shape[0] > 0:
                # Get the index of the next change in signal type
                next_change_cidx = all_trial_changes[next_change[0]]
                # Get the new post-change signal type
                changeto = raster_mat[idx_curr_r, next_change_cidx+1]
                # Delete the indices of this change to avoid it being counted double
                idx_c = np.delete(idx_c, next_change[0], axis=0)
                idx_r = np.delete(idx_r, next_change[0], axis=0)
                
        # Only if there is an actual change in the signal type, store the change
        if changefrom != changeto: 
            # If there was an overlap (type above max in last matrix)
            if changefrom > len(raster_mats):
                # Look up the conversion to replace the overlap type by the two underlying signal types
                changefrom = overlap_conversion[changefrom]
            # Repeat for the post-change types
            if changeto > len(raster_mats):
                changeto = overlap_conversion[changeto]
            
            # Append the changes (depending on the number of entries, two for overlaps)
            if changefrom.size > 1 and changeto.size > 1:
                # Case: overlap prior to and after change
                changes.append((changefrom[0], changeto[0]))
                changes.append((changefrom[1], changeto[1]))
                # Store the indices of the change
                indices.append((idx_curr_r, idx_curr_c))
                indices.append((idx_curr_r, idx_curr_c))
            elif changefrom.size > 1:
                # Case: overlap prior to change
                changes.append((changefrom[0], changeto))
                changes.append((changefrom[1], changeto))
                # Store the indices of the change
                indices.append((idx_curr_r, idx_curr_c))
                indices.append((idx_curr_r, idx_curr_c))
            elif changeto.size > 1:
                # Case: overlap after change
                changes.append((changefrom, changeto[0]))
                changes.append((changefrom, changeto[1]))
                # Store the indices of the change
                indices.append((idx_curr_r, idx_curr_c))
                indices.append((idx_curr_r, idx_curr_c))
            else:
                # Case: no overlaps
                changes.append((changefrom, changeto))
                # Store the indices of the change
                indices.append((idx_curr_r, idx_curr_c))

        # Remove the indices that were delt with in the current iteration
        idx_r = np.delete(idx_r, 0, axis=0)
        idx_c = np.delete(idx_c, 0, axis=0)
    
    return np.array(changes), np.array(indices)



def count_signal_switches_exp_wise(raster_mats:list[np.ndarray], time_vec:np.ndarray, poss_evt_changes:np.ndarray, trial_nums:list[int], bin_width:float, 
                                 stim_len:float=4.0, tolerance:float=0.1, norm:bool=True) -> tuple[np.ndarray, np.ndarray]:
    """### Find all changes of signal type from raster matrices (split by experiments)
    This is a wrapper around 'get_signal_switches' to separate the results conveniently for single experiments which are combined in the 
    raster matrices. The possible signal changes are counted for each experiment. If requested those counts can be transformed to 
    probabilities by dividing them by the product of bins and trials, i.e. the number of possibilities for a change to occurr. 
    
    Args:
        raster_mats (list[n*m array]): list of zero-non-zero matrices indicating the times of events in n trials over m sampling points
        time_vec (m*1 array): time vector corresponding to axis 1 of the raster matrices
        poss_evt_changes (k*1 array[tuple]): all signal type changes that are possible/that should be counted
        trial_nums (list): number of trials in each experiment
        bin_width (float): bin width that was used to creat the raster matrices [s]
        stim_len (float): length of the stimulation, time in time_vec where the stimulus ends [s]
        tolerance (float): time between two signals that should not be considered [s], default: 0.1 s
        norm (bool): state whether or not to normalize the counts of possible changes to counts per trial and second

    Returns:
        tuple: with two items
            - array (n*k): (normalized) counts for the k possible signal type switches in n experiments prior to stimulation
            - array (n*k): (normalized) counts for the k possible signal type switches in n experiments during stimulation
    """
    # Allocate lists to collect the signal type changes
    changes_pre = []
    changes_dur = []

    # Define indices for the pre-stimulus time and time during stimulation
    pre_stim_idx = time_vec < 0
    dur_stim_idx = np.logical_and(time_vec >= 0, time_vec <= stim_len)

    # Set the initial trail to start with
    fromtrial = 0
    # Iterate over all experiments (length of stim_volts as a proxy for the number of experiments)
    for ntrials in trial_nums:
        # Set upper row index for the current experiment
        totrial = fromtrial + ntrials
        
        # Get the signal type changes prior to stimulation
        chg_mf, _ = get_signal_switches([raster_mats[0][fromtrial:totrial, pre_stim_idx],
                                         raster_mats[1][fromtrial:totrial, pre_stim_idx],
                                         raster_mats[2][fromtrial:totrial, pre_stim_idx]], 
                                         bin_width, tolerance)
        # Count how many changes of which kind were found
        changes_pre.append([ np.sum(np.sum(chg_mf == chg, axis=1) == 2) if chg_mf.size > 0 else 0
                            for chg in poss_evt_changes ])

        # Get the signal type changes during stimulation
        chg_mf, _ = get_signal_switches([raster_mats[0][fromtrial:totrial, dur_stim_idx],
                                         raster_mats[1][fromtrial:totrial, dur_stim_idx],
                                         raster_mats[2][fromtrial:totrial, dur_stim_idx]], 
                                         bin_width, tolerance)
        # Count the occurrances of each kind of change
        changes_dur.append([ np.sum(np.sum(chg_mf == chg, axis=1) == 2) if chg_mf.size > 0 else 0
                            for chg in poss_evt_changes ])

        # Increase the lower row index for the next iteration
        fromtrial += ntrials
    
    # Transform lists to arrays
    changes_pre = np.array(changes_pre)
    changes_dur = np.array(changes_dur)

    # If requested convert the counts 
    if norm:
        # Calculate stimulus and pre-stimulus window length
        stim_len = np.sum(dur_stim_idx) * bin_width
        pre_stim_len = np.sum(pre_stim_idx) * bin_width

        # Calculate division array to normalize the signal change counts 
        # (length of counting period * number of trials)
        division_arr_dur = np.array([ stim_len * ntrials for ntrials in trial_nums ])
        division_arr_pre = np.array([ pre_stim_len * ntrials for ntrials in trial_nums ])

        # Normalize the switch counts [#/(trial*s)]
        cnt_changes_dur_norm = np.vstack([ changes_dur[:,clmn] / division_arr_dur for clmn in range(changes_dur.shape[1]) ]).T
        cnt_changes_pre_norm = np.vstack([ changes_pre[:,clmn] / division_arr_pre for clmn in range(changes_pre.shape[1]) ]).T

        return cnt_changes_pre_norm, cnt_changes_dur_norm
    else:
        return changes_pre, changes_dur



def contdata_fi_curve(data:np.ndarray, time_vec:np.ndarray, stim_volts:np.ndarray, stim_period:list[float]=[0,4], 
                      trial_nums:list[int]|None=None, relative:bool=True, median:bool=True) -> np.ndarray:
    """### Calculate average responses per experiment and stimulus intensity 

    Args:
        data (array n*m): input data organized as as n trials of m data points all aligned to the time vector along axis 1 (m), axis 0 must contain the experiment order
        time_vec (array m*1): time vector corresponding to axis 1 of data [s]
        stim_period (list): start and stop value of the stimulation period, default: [0,4] [s]
        trial_nums (list|None): list of trial numbers in each of k experiments along axis 0 of data, if None all trials are used at once, default: None
        relative (bool): whether or not to relativize the values during stimulation using the values before stimulation, default: True
        median (bool): whether or not to calculate the median over time, if False the mean is calculated, default: True

    Returns:
        array (shape k*1): which contains the averaged data (across time and trials) for all k experiments.   
    """
    # If no trial numbers are given, take the whole input data at once
    if trial_nums == None:
        trial_nums = [data.shape[0]]

    # Create index vectors for pre-stimulus and stimulus time
    pre_stim_sel = time_vec < stim_period[0]
    stim_sel = np.logical_and(time_vec >= stim_period[0], time_vec < stim_period[-1])
    # Define to row index variables for trial selection
    start_idx = 0
    stop_idx = 0
    # List to collect results
    fi_data = []

    stim_volts_unique = np.unique(stim_volts)

    # Iterate over the given trial numbers
    for i in trial_nums:
        # Set the stop index to i (number of trials for the first experiment)
        stop_idx += i

        # Check if stop index is still in the suitable range, if not set to maximal index
        if stop_idx >= data.shape[0]:
            stop_idx = data.shape[0] - 1

        # Subset the data for the current trial (experiment)
        stim_volts_trial = stim_volts[start_idx:stop_idx] 
        stim_data_trial = data[start_idx:stop_idx, stim_sel]
        pre_data_trial = data[start_idx:stop_idx, pre_stim_sel]
        
        # If results should be relative
        if relative:
            # Calculate difference (during-before stimulation) of the averages/medians over trials and over time 
            if median:
                fi_data.append([ np.median(np.nanmedian(stim_data_trial[stim_volts_trial==stv, :], axis=1) - 
                                           np.nanmedian(pre_data_trial[stim_volts_trial==stv, :], axis=1)) for stv in stim_volts_unique ])
            else:
                fi_data.append([ np.mean(np.nanmean(stim_data_trial[stim_volts_trial==stv, :], axis=1) - 
                                         np.nanmean(pre_data_trial[stim_volts_trial==stv, :], axis=1)) for stv in stim_volts_unique ])
        else:
            # If results should not be relative, calculate the mean/median over trials and time for the given stimulus period
            if median:
                fi_data.append([ np.median(np.nanmedian(stim_data_trial[stim_volts_trial==stv, :], axis=1)) for stv in stim_volts_unique ])
            else:
                fi_data.append([ np.mean(np.nanmean(stim_data_trial[stim_volts_trial==stv, :], axis=1)) for stv in stim_volts_unique ])

        # Increase the start index for the nex iteration
        start_idx += i

    return np.array(fi_data)

