""" # Functions for plotting and modifying plot appearance
This module includes functions to prepare, create and modify plots. 

Following functions are included:
 - align_zeros:               align the zero on the y axis across multiple axes
 - equal_axlims:              make the axis limits of multiple axes equal
 - add_stim_bar:              add a horizontal bar on top of a plot to indicate a stimulation time period
 - add_condition_pictogram:   add a pictogram on top of a plot to indicate the condition the data belong to
 - add_significance_stars:    add stars on top of a plot (e.g. boxplot) to indicate significant results
 - axis_to_scale:             remove an axis with ticks to replace it with a scale bar
 - ax_colorcode:              color-code the tick labels of an axis
 - iei_distro:                create a histogram of inter-event-intervals
 - boxplot_dp:                create custom boxplots with the option to overlay the data points
 - psth_plt:                  create PSTHs from matrices
 - raster_plot:               create a raster plot from a matrix
 - violinplot:                create a custom violin plot
 - prep_sankey_from_mat:      prepare data for plotting in a plotly sankey diagram from a raster matrix

Useful links:
 - [matplotlib named colors](https://matplotlib.org/stable/gallery/color/named_colors.html#css-colors)
 - [matplotlib color maps](https://matplotlib.org/stable/users/explain/colors/colormaps.html#sequential)

Author:         Bjarne Schultze <br>
Last modified:  02.12.2024
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_rgba, to_hex
from matplotlib.offsetbox import (AnchoredOffsetbox,VPacker,TextArea)
from matplotlib.typing import ColorType
import modules.analysis_utils as utils


def align_zeros(ax) -> None:
    """### Align the zeros on mulitple y axis
    Modified after: https://stackoverflow.com/a/68869054 ('Hyde Fukui', 2021) [last accessed; 08.07.2024]

    Args:   
        ax (axes handle): matplotlib axes handle for the axes to modify

    Returns:
        None
    """
    # Allocate empty dicts to collect calculations
    yrange = {}     #  ymax - ymin for for current limits
    yratios = {}    #  ratio of lower limit to range

    for cax in ax:
        # Get current limits
        ylims = list(cax.get_ylim())
        # Calculate range 
        yrange[cax] = ylims[1] - ylims[0]
        # Calculate ratio of lower limit to whole range
        yratios[cax] = -ylims[0]/yrange[cax]
    
    for cax in ax:
        # Find new ylim values
        ylims_new = []
        ylims_new.append(min(-yrange[cax] * np.array(list(yratios.values()))))       # Lower limit
        ylims_new.append(max(yrange[cax] * (1-np.array(list(yratios.values())))))    # Upper limit
        
        # Set new axes limits
        cax.set_ylim(ymin=ylims_new[0], ymax=ylims_new[1])

        

def equal_axlims(ax:list, axis:str="y") -> None:
    """### Make the axes limits equal among a given set of axes

    Args:
        ax (list of axes handles): matplotlib axes handles for the axes to modify
        axis (str): states the axis to be modified, "x" or "y", default: "y"
    
    Returns:
        None    
    """
    if axis == "y":
        # Get the current y limits for all given axes
        curr_ylims = [ cax.get_ylim() for cax in ax ]
        # Set the new limits to the overall minimum and maximum
        new_ylims = (np.min(curr_ylims), np.max(curr_ylims))
        # Apply new limits to all axes
        [ cax.set_ylim(new_ylims) for cax in ax ]
    elif axis == "x":
        # Get the current x limits for all given axes
        curr_xlims = [ cax.get_xlim() for cax in ax ]
        # Set the new limits to the overall minimum and maximum
        new_xlims = (np.min(curr_xlims), np.max(curr_xlims))
        # Apply new limits to all axes
        [ cax.set_xlim(new_xlims) for cax in ax ]



def add_stim_bar(ax, stim_period:list[float]=[0.0,4.0], stim_col:ColorType="#DB0D55") -> None:
    """### Add a bar to indicate the time of stimulation

    Args:
        ax (axes handle): handle to the axes on which to add the bar
        stim_period (list[float]): two time points to indicate start and stop of the stimulation period, default: [0.0, 4.0]
        stim_col (ColorType): color of the bar, default: #DB0D55 (red)

    Returns:
        None    
    """
    # Check if one or multiple axes
    if isinstance(ax, np.ndarray):
        # If multiple, iterate other the axes handles
        for cax in ax:
            # Get current y limits
            ylim_val = cax.get_ylim()
            # Set a y value for plotting the bar
            ylim_bar = ylim_val[-1]  + ylim_val[-1] * 0.015
            # Color a bar-like region to during the stim_period
            cax.fill_between(stim_period, ylim_bar, ylim_bar+ylim_bar*0.025, color=stim_col, clip_on=False)
            # Adapt the y limits
            cax.set_ylim(ylim_val)
    else:
        # In case of a single axes handle
        ylim_val = ax.get_ylim()
        ylim_bar = ylim_val[-1]  + ylim_val[-1] * 0.01
        ax.fill_between(stim_period, ylim_bar, ylim_bar+ylim_bar*0.02, color=stim_col, clip_on=False)
        ax.set_ylim(ylim_val)



def add_condition_pictogram(ax, cond:str, text_col:ColorType="#000000", circle_col:ColorType="#DB0D55", 
                            markersize:float=22, fontsize:float=12, **kwarg) -> None:
    """### Add little pictograms to indicate male or male-femal condition

    Args:
        ax (axes handle): handle to the axes on which to add the pictograms
        cond (str): indicate the condition, "m" for male symbol, "mf" for male and female symbol, if no one of those, the given text  
                    will be used
        text_col (ColorType): color of the text or symbols, default: black
        circle_col (ColorType): color of the circle, default: #DB0D55 (red tone)
        markersize (float): size of the circle, based on the matplotlib markersize, default: 22
        fontsize (float): font size for the symbols or the text which is placed in the circle, default: 12
        **kwarg:
         - x (float, optional): x coordinate of the circle-center, if not given circle is placed automatically
         - y (float, optional): y coordinate of the circle-center, if not given circle is placed automatically
    
    Returns:
        None
    """
    # Get x coordinate for the circle from input if given
    if "x" in kwarg:
        xcenter = kwarg["x"]
    else:
        # Get the x limits of the axes
        x_lim = ax.get_xlim()
        # Define a x value for the center of the circle
        xcenter = x_lim[-1] - np.abs(np.diff(x_lim)) * 0.08
    # Get x coordinate for the circle from input if given
    if "y" in kwarg:
        ycenter = kwarg["y"]
    else:
        # Get the y limits of the axes
        y_lim = ax.get_ylim()
        # Define a y value for the center of the circle
        ycenter = y_lim[-1] - np.abs(np.diff(y_lim)) * 0.1

    # Add a big circle marker in the upper right corner
    ax.plot(xcenter, ycenter, 'o', markersize=markersize, alpha=0.7, color=circle_col, mew=1.5, clip_on=False)
    # Write theier the male or the male and female symbol in the marker
    if cond == "m":
        ax.text(xcenter, ycenter, "$\u2642$", color=text_col, 
                horizontalalignment="center", verticalalignment="center", fontsize=fontsize)
    elif cond == "mf":
        ax.text(xcenter, ycenter, "$\u2642\u2640$", color=text_col, 
                horizontalalignment="center", verticalalignment="center", fontsize=fontsize)
    else: 
        ax.text(xcenter, ycenter, cond, color=text_col, 
                horizontalalignment="center", verticalalignment="center", fontsize=fontsize)



def add_significance_stars(ax, effect:list[bool], positions:list[float]|np.ndarray, width:float=0.2, color:ColorType="black", 
                           line_ends:bool=False, textpad=0.1) -> None:
    """### Adds stars to indicate significant effect (e.g. above boxplots)

    Args:
        ax (axes handle): matplotlib axes handle for the axes to modify
        effect (list[bool]): indicates whether there was a significant effect (True) or not (False)
        positions (list): x positions at which to place the (not-) significant markers
        width (float): width of the line underneath the (not-) significant markers, default: 0.2
        color (ColorType): color of the lines and stars/markers, matplotlib-compatible color specification, default: "black"
        
    Returns:
        None
    """
    # Check if line ends pointing down are requested
    if line_ends:
        marker = 3
    else:
        marker = None

    # Get the current y limits of the axis
    curr_ylim = ax.get_ylim()
    # Define y a new upper y limit and positions for the lines and stars
    ymax = curr_ylim[-1] + np.diff(curr_ylim) * 0.15
    yline = curr_ylim[-1] + np.diff(curr_ylim) * 0.05
    ystars = curr_ylim[-1] + np.diff(curr_ylim) * textpad
    yns = curr_ylim[-1] + np.diff(curr_ylim) * (textpad-0.02)

    # Iterate over the effect and positions arrays
    for ef,pos in zip(effect, positions):
        # Add a line with the given width
        ax.plot([pos-width/2, pos+width/2], [yline, yline], color=color, marker=marker, mew=1.5, ms=4)
        # If there was an effect, add an aterisk, otherwise add a short line
        if ef:
            ax.plot(pos, ystars, color=color, marker=(6,2,0), markersize=7)
        else:
            ax.text(pos, yns, "n.s.", horizontalalignment="center")
    
    # Adapt the limits of the y axis
    ax.set_ylim(bottom=curr_ylim[0], top=ymax)



def axis_to_scale(ax, scale_unit:str|None, scale_len:float=1, axis="x", linewidth:float=4, txt_offset:float|str="auto", **kwarg)-> None:
    """ ### Replace an axis with ticks by a scale bar
    This function takes an axes handle and removes the either the left or bottom spine and adds a scale bar instead. The scale bar is 
    positioned vertical at the lower xlim value for the y axis or horizontally at the lower ylim value for the x axis. 

    Args:
        ax (axes handle): handle to the matplotlib axes which should be modified
        scale_unit (str|None): give the unit of the scale bar which is placed beneath/left to the scale bar and its length, if `None`
                               the scale bar will not be labelled
        scale_len (float): length of the scale bar in axes coordinates, default: 1
        axis (str): indicates which axis to modify, "x" or "y", default: "x"
        linewidth (float): line width of the scale bar, default: 4
        txt_offset (float|str): 
        **kwarg: further arguments to be passed to ax.text(), e.g. `fontsize`, `color`, ..., `horizontalalignment` and `verticalalignment` are set
    
    Returns:
        None    
    """
    # Get the axes limits
    ax_xlim = ax.get_xlim()
    ax_ylim = ax.get_ylim()

    # Set text offset (from line)
    if txt_offset == "auto":
        if axis == "x":     txt_offset = (np.diff(ax_ylim)*0.003)[0]
        elif axis == "y":   txt_offset = (np.diff(ax_xlim)*0.001)[0]


    if axis == "x":    
        # Add a scale bar and a text stating its length
        ax.plot([ax_xlim[-1]-scale_len, ax_xlim[-1]], [ax_ylim[0]-np.diff(ax_ylim)*0.002, ax_ylim[0]-np.diff(ax_ylim)*0.002], 
                color="k", linewidth=linewidth)
        if scale_unit != None:
            ax.text(ax_xlim[-1]-0.5*scale_len, ax_ylim[0]-txt_offset, f"{scale_len} {scale_unit}", horizontalalignment="center", 
                    verticalalignment="top", **kwarg)

        # Remove the spines and tick values
        ax.spines["bottom"].set_visible(False)
        ax.set_xticks([])
    elif axis == "y":
        # Add a scale bar and a text stating its length
        ax.plot([ax_xlim[0], ax_xlim[0]], [ax_ylim[0], ax_ylim[0]+scale_len], color="k", linewidth=linewidth)
        if scale_unit != None:
            ax.text(ax_xlim[0]-np.diff(ax_xlim)*0.001, ax_ylim[0]+0.5*scale_len, f"{scale_len} {scale_unit}", horizontalalignment="right", 
                    verticalalignment="center", **kwarg)

        # Remove the spines and tick values
        ax.spines["left"].set_visible(False)
        ax.set_yticks([])



def ax_colorcode(ax, color_dict:dict, axis:str="both", legend:bool=False, **kwarg) -> None:
    """### Color code the tick labels of an axes showing an imshow or pcolormesh plot

    Args:
        ax (axes handle): axes containing the imshow/pcolormesh plot
        color_dict (dict): dictionary mapping axes labels (keys) to color values 
        axis (str): specify which axis to modify ("x"|"y"|"both"), default: "both"
        legend (bool): indicating if a legend should be displayed, default: False
        **kwarg:
         - color_names (dict, optional): required if legend=True, dictionary mapping legend entries (keys) to color values
         - l_title (str, optional): title for the legend

    Returns:
        None
    """
    # Color code x axis labels
    if axis == "both" or axis == "x":
        # For each x label 
        for label in ax.get_xticklabels():
            # Check if it is in the color dict
            if label.get_text() in color_dict.keys():
                # If so change its color according to the dict
                label.set_color(color_dict[label.get_text()])

    # Color code y axis labels
    if axis == "both" or axis == "y":
        # For each x label 
        for label in ax.get_yticklabels():
            # Check if it is in the color dict
            if label.get_text() in color_dict.keys():
                # If so change its color according to the dict
                label.set_color(color_dict[label.get_text()])

    # If requested, add a lengend for the tick colors
    if legend and "color_names" in kwarg:
        # Get the color name dict
        color_names = kwarg["color_names"]

        # Create text boxes with the c_names written in the respective color
        boxes = []
        if "l_title" in kwarg:
            # Add the title
            boxes.append(TextArea(kwarg["l_title"], textprops=dict(color="k", fontweight="bold", fontsize=12)))

        # Create one entry for each color using the given names
        for coln in color_names:
            boxes.append(TextArea(coln, textprops=dict(color=color_names[coln], fontsize=12)))

        # Stack the text boxes in one box
        box = VPacker(children=boxes, align="left", pad=0, sep=5)

        # Create legend box 
        anchored_box = AnchoredOffsetbox(loc='upper left', child=box, pad=0.5, frameon=True,
                                        bbox_to_anchor=(1.03, 1), bbox_transform=ax.transAxes, borderpad=0)
        # Add the legend to the right subplot
        ax.add_artist(anchored_box)
    elif legend:
        # If no color names were given, print a message 
        print("Can't create a legend without names for the colors!")



def iei_distro(event_times:list|np.ndarray, cutoff:float, **kwarg) -> np.ndarray:
    """### Calculate and plot the distribution of inter-event-intervals

    Args:
        event_times (list|array): list/array with the event times [s]
        cutoff (float): cutoff for the maximal inter-event-interval [ms], only values below cutoff are plotted
        **kwarg:
         - ax (axes handle): handle for the axes on which to plot

    Retruns:
        np.ndarray: inter-event-intervals considering the cutoff    
    """
    # Calculate the inter-event-interval
    ieinterval = np.abs(np.diff(event_times) * 1000)    # [ms]
    ieinterval = ieinterval[ieinterval < cutoff]

    # Create histogram either in provided axis or in new plot
    if "ax" in kwarg:
        kwarg["ax"].hist(ieinterval, bins=20, density=True)
    else:
        plt.hist(ieinterval, bins=20, density=True)
    
    return ieinterval



def boxplot_dp(data, *, labels:list[str]|None=None, boxcolor:list[ColorType]|ColorType="#5AC1DB", plot_dp:bool=True, markercolor:ColorType="#146AC7", 
               boxalpha:float=0.7, markeralpha:float=1.0, markersize:float=5, std_dp:float=0.05, **kwarg) -> None:
    """### Create custom boxplots with the option to overlay data points
    
    Args:
        data (list[array]|array): data to plot, each list entry or each slice along axis 1 of an array is plotted in a separate boxplot
        lables (list[str]|None): a list of labels for the single boxplots, if None no lables will be added, default: None
        boxcolor (list[ColorType]|ColorType): either one color for all boxplots or a list with one color for each boxplot, default: "#5AC1DB" (blue tone)
        plot_dp (bool): indicate whether or not to plot the data points, default: True
        markercolor (ColorType): color of the data points when overlayed, default: "#146AC7 (green tone)
        boxalpha (float): alpha value for the boxplot boxes (0-1), default: 0.7
        markeralpha (float): alpha vlaue for the data points if overlayed (0-1), default: 1.0
        markersize (float): size of the markers if overlayed, default: 5
        std_dp (float): standard deviation of the random offset of the data points, default: 0.05
        **kwarg:
         - ax (axes handle): handle for the axes to plot on
         - boxwidth (float): width of the boxplot boxes
    
    Returns:
        None
    """
    # Create histogram either in provided axis or in new plot
    if "ax" in kwarg:
        ax = kwarg["ax"]
    else:
        _, ax = plt.subplots(1, 1)
    
    # Check if boxwidth was provided, if so read the value if not use default
    if "boxwidth" in kwarg:
        box_width = kwarg["boxwidth"]
    else:
        box_width = 0.25

    # Get a random number generator to generate random offsets
    rng = np.random.default_rng(seed=50667)
    # Plot the data ax boxplots
    bxplt = ax.boxplot(data, patch_artist=True, tick_labels=labels, widths=box_width)

    # If requested, plot the data points
    if plot_dp:
        # Distinguish different input data types
        if isinstance(data, list) and len(data[0]) > 1:
            for idx,grp in enumerate(data):
                x = np.repeat(idx+1, len(grp)) + rng.normal(0, std_dp, len(grp))
                ax.plot(x, grp, "o", markersize=markersize, color=markercolor, alpha=markeralpha)
        elif isinstance(data, np.ndarray) and data.ndim > 1:
            for idx in range(data.shape[1]):
                x = np.repeat(idx+1, data.shape[0]) + rng.normal(0, std_dp, data.shape[0])
                ax.plot(x, data[:, idx], "o", markersize=markersize, color=markercolor, alpha=markeralpha)
        else:
            x = np.repeat(1, len(data)) + rng.normal(0, std_dp, len(data))
            ax.plot(x, data, "o", markersize=markersize, color=markercolor, alpha=markeralpha)

    # Color the boxes
    if isinstance(boxcolor, list):
        for patch,col in zip(bxplt["boxes"],boxcolor):
            patch.set_facecolor(col)
            patch.set_alpha(boxalpha)  
    else:
        for patch in bxplt["boxes"]:
            patch.set_facecolor(boxcolor)
            patch.set_alpha(boxalpha)
    


def psth_plot(time_vec:np.ndarray, hists:list[np.ndarray], colors:list[ColorType], labels:list[str], 
              stim_period:list|tuple|np.ndarray=[0,4], stim_col:ColorType='#DB0D55', **kwarg) -> None:
    """### Create a PSTH plot from multiple PSTH lines

    Args: 
        time_vec (array): time vector corresponding to the PSTH lines in `hist`
        hists (list[array]): list of arrays holding PSTH lines
        colors (list[ColorType]): list of colors (one for each PSTH line in `hist`)
        labels (list[str]): labels for the PSTHs (one for each PSTH line in `hist`)
        stim_period (list|tuple|array): start and stop value of the stimulation period
        stim_col (ColorType): color to use for a bar indicating the stimulation period
        **kwarg:
         - legend_off (bool, optional): indicate whether or not to turn off the legend, default: False
         - stim_bar (bool, optional): indicate whether or no to plot a bar indicating the stimulation time
         - ax (axes handle, optional): axes handle for the axes on which to place the plot, if not given, a new figure is created
    
    Returns:
        None
    """

    # Create plot either in provided axis or in new plot
    if "ax" in kwarg:
        ax = kwarg["ax"]
    else:
        _, ax = plt.subplots(1, 1)
    
    # Plot the PSTHs
    for hist, col, label in zip(hists, colors, labels):
        ax.plot(time_vec, hist, label=label, color=col)

    # Add a bar to show the stimulation period if not tured off
    if "stim_bar" not in kwarg or kwarg["stim_bar"]:
        ylim_val = ax.get_ylim()
        ylim_bar = ylim_val[-1]  + 0.01
        ax.fill_between(stim_period, ylim_bar, ylim_bar+ylim_bar*0.03, color=stim_col, clip_on=False)
        ax.set_ylim(ylim_val)

    # Add legend if not explicitly turned off
    if "legend_off" not in kwarg or not kwarg["legend_off"]:
        ax.legend()
 


def raster_plot(raster_mats:list[np.ndarray], raster_time:np.ndarray, stim_volts:np.ndarray, labels:list[str], colors:list[ColorType], 
                stim_period:list|tuple|np.ndarray=[0,4], stim_col:ColorType='#DB0D55', alpha:float=0.7, **kwarg) -> None:
    """### Create a raster plot from matrices

    Args:
        raster_mats (list[array]): raster matrices for different signal types (one matirx per type), a signal should be encoded as non-zero values, 
                                   no signal should be encoded as 0
        raster_time (array): time vector for the raster matrices, corresponding to axis 1 of the matrices
        stim_volts (array): the stimulation voltages for each trial, corresponding to axis 0 of the matrices
        labels (list[str]): labels for the signal types (one per matrix)
        colors (ColorType): colors for the different signal types (one per matrix)
        stim_period (list|tuple|array): start and stop value of the stimulation period
        stim_col (ColorType): color to use for a bar indicating the stimulation period
        **kwarg:
         - legend_off (bool, optional): indicate whether or not to turn off the legend, default: False
         - stim_watts (list|array, optional): if given, the simulus values are changed to the given ones
         - stim_bar (bool, optional): indicate whether or no to plot a bar indicating the stimulation time
         - ax (axes handle, optional): axes handle for the axes on which to place the plot, if not given, a new figure is created
    
    Returns: 
        None
    """
    # Create histogram either in provided axis or in new plot
    if "ax" in kwarg:
        ax = kwarg["ax"]
    else:
        _, ax = plt.subplots(1, 1)

    # Convert matplotlib color names or hex values to rgba values (including transparency )
    colors = [ to_rgba(col, alpha) for col in colors ]

    # Sort all stimuli according to intensity, obtain sort index
    stim_sort_idx = np.argsort(stim_volts) 
    stim_volt_srt = stim_volts[stim_sort_idx]

    # Iterate over the list of raster matrics
    for iraster, raster in enumerate(raster_mats):
        # Sort the current raster matrix
        raster_srt = raster[stim_sort_idx, :]
        # Plot the sorted raster matrix
        ax.pcolormesh(raster_time, np.arange(0, raster_srt.shape[0], 1), raster_srt, 
                      cmap=ListedColormap([[1,1,1,0], colors[iraster]]))

    # Adjust the y limit
    ax.set_ylim(0, raster_mats[0].shape[0])

    # Add legend if not explicitly turned off
    if "legend_off" not in kwarg or not kwarg["legend_off"]:
        for col,label in zip(colors, labels):
            ax.plot(1, -1, color=col, label=label)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2,1.01))

    # Find indices of changes in the stimulus intensities
    stim_change = np.where(np.diff(stim_volt_srt) > 0) + np.array(1)
    stim_change = np.insert(stim_change, 0, 0)
    # Use the points where stimulus changed as y ticks
    if "stim_watts" in kwarg:
        ax.set_yticks(stim_change, kwarg["stim_watts"])
    else:
        ax.set_yticks(stim_change, stim_volt_srt[stim_change])

    # Add a bar to show the stimulation period if not tured off
    if "stim_bar" in kwarg and kwarg["stim_bar"]:
        ylim_bar = ax.get_ylim()[-1]  + np.diff(ax.get_ylim()) * 0.05
        ax.fill_between(stim_period, ylim_bar, ylim_bar+ylim_bar*0.03, color=stim_col, clip_on=False)



def violinplot(data:list[np.ndarray]|np.ndarray, *, labels:list[str], positions:list[float]|None=None, colors:list[ColorType]|ColorType="#5AC1DB", width:float=0.5,
               alpha:float=0.7, plot_dp:bool=False, markeralpha:float=1.0, markersize:float=5.0, markercolor:str|list[float]|tuple[float]="#146AC7", **kwarg) -> None:
    """### Create a custom violin plot for one or multiple datasets

    Args:
        data (list[np.ndarray]|np.ndarray): data to be plotted as violin plots
        labels (list[str]): labels for the x axis (one per violin plot)
        positions (list|None): x positions on which to place the violin plots, default: None (places plots at  1, 2, 3, ...)
        colors (str|tuple|list): colors for the violin plots, all matplotlib color specifications possible, default: "#5AC1DB"
        width (float): set the width/maximal y-expansion of the violin plots, default: 0.5
        alpha (float, ]0;1[): set the alpha value for the plots, default: 0.7
        plot_dp (bool): set whether or not to plot the data points behind the violin plots, default: False
        markeralpha (float): alpha value for the data points if plotted, default: 1.0
        markersize (float): size of the data points if plotted, default: 5.0
        markercolor (str|tuple|list): color for the data points if plotted, all matplotlib color specifications possible, default: "#146AC7"
        **kwarg:
         - ax (matplotlib axes handle): axes handle for the axes to place the plot, if not provided new plot is created
    
    Returns:
        None
    """
    # Create histogram either in provided axis or in new plot
    if "ax" in kwarg:
        ax = kwarg["ax"]
    else:
        _, ax = plt.subplots(1, 1)

    # If data is a list ...
    if isinstance(data, list):
        # Check for empty arrays in data
        nempty_data = [ False if dat.shape[0] == 0 else True for dat in data ]
        # Select only non-empty data arrays
        data = [ dat for dat,empt in zip(data,nempty_data) if empt ]
        # If positions are given, select those corresponding to non-empty data arrays 
        if positions != None: 
            plt_pos = [ pos for pos,empt in zip(positions,nempty_data) if empt ]
        # If no positions were given, select labels corresponding to non-empty data arrays. Use all given positions
        else:
            labels = [ la for la,empt in zip(labels,nempty_data) if empt ]
            plt_pos = positions
    # If only one data array is given ...
    else:
        # Raise an error if the data array is empty
        if data.shape[0] == 0: raise ValueError("Cannot create violin plot from empty array!")
        # If it is not empty, use all positions
        plt_pos = positions

    # Determine the number of violin plots to create for axes scaling
    if isinstance(data, list): 
        dat_len = len(data)
    else: 
        dat_len = 1

    # Create the violin plots (with median lines)
    vioplt = ax.violinplot(data, positions=plt_pos, showmedians=True, widths=width)
    
    # Set the given labels as xticks
    if positions == None:
        ax.set_xlim((1-width-0.05, dat_len+width+0.05))
        xticks = list(range(1, dat_len+1))
        ax.set_xticks(xticks, labels)
    else:
        ax.set_xlim((positions[0]-width-0.05, positions[-1]+width+0.05))
        ax.set_xticks(positions, labels)

    # Adapt the style of the plot
    # Get the minima and maxima lines
    cmaxes = vioplt["cmaxes"].get_segments() # type: ignore
    cmins = vioplt["cmins"].get_segments() # type: ignore

    # Iterate over the violin plots to adapt the style
    for datidx in range(dat_len):
        if isinstance(colors, list):
            col = colors[datidx]
        else:
            col = colors
        # Set color and alpha of the body
        vioplt["bodies"][datidx].set_color(col) # type: ignore
        vioplt["bodies"][datidx].set_alpha(alpha) # type: ignore
        # Set the width of the min and may lines
        cmaxes[datidx][:,0] += np.array([0.075,-0.075])
        cmins[datidx][:,0] += np.array([0.075,-0.075])

    # Apply the changes to the minima and maxima lines
    vioplt["cmaxes"].set_segments(cmaxes) # type: ignore
    vioplt["cmins"].set_segments(cmins) # type: ignore
    # Set the color for all lines
    vioplt["cmaxes"].set_colors(colors) # type: ignore
    vioplt["cmins"].set_colors(colors) # type: ignore
    vioplt["cbars"].set_colors(colors) # type: ignore
    vioplt["cmedians"].set_colors(colors) # type: ignore
    # Increase line width for median line
    vioplt["cmedians"].set_linewidth(2.0)

    # If requested, plot the data points
    if plot_dp:
        # Define a random-number-generator
        rng = np.random.default_rng(seed=50667)
        # Iterate over the data arrays
        for idx,grp in enumerate(data):
            # Create a vector of random x offsets for each data point
            x = np.repeat(idx+1, len(grp)) + rng.normal(0, 0.05, len(grp))
            # Plot the data points behind the violin plot
            ax.plot(x, grp, "o", markersize=markersize, color=markercolor, alpha=markeralpha, zorder=-1)



def prep_sankey_from_mat(raster_mats:list[np.ndarray], time_vec:np.ndarray, colors:list[ColorType], stim_len:float=4, pre_stim_window:float=0.05, 
                         alpha:float=0.7, sampling_rate:float=10000) -> tuple[np.ndarray, list[float], list[str], list[str]]:
    """### Calculate the data for a sankey diagram from a list of matrices
    Each event type should be given in a separate matrix where no events are encoded as 0 and events as numbers. In the 
    first matrix events should be ones, in the second it should be twos, in the third threes ... The function then finds
    the changes in the event types and counts them. It also creates a suitable list of colors for the boxes and links
    of the sankey diagram. This is adjusted to the input requirements of a plotly sankey diagram. 

    Args:
        raster_mats (list[array]): matrices for different event types, trials along axis 0, time along axis 1
        time_vec (array): time vector corresponding to axis 1 of the matrices
        colors (list): list of colors corresponding to the different event types, length must match len(raster_mats)+1, 
                       first entry for no signal, any matplotlib color specification possible
        stim_len (float): length of the stimulation, time in time_vec where the stimulus ends [s]
        pre_stim_window (float): length of the window used to search for non-zero singal types before stimulation [s], default 0.05 
        alpha (float): alpha value for the links of the sankey diagram, default: 0.7
        sampling_rate (float): sampling rate for the raster matrices

    Returns:
        tuple: 1. item - an array (n*2) which contains the n possible event type changes, per row change from first to second column. 
        2. item - a list (n entries) with the number of occurrances of each possible event type change. 
        3. item - a list (2*len(raster_mats) entries) with colors for the boxes in the sankey diagram, same color twice for left and right side of the diagram
        4. item - a list (n entries) with colors for each of the possible event type changes, color is determined by the source
    """
    # Sum the raster matrices for all signal types (4 translates to an overlap of pulse and vibration, 5 denotes an overlap of sine and vibration)
    raster_mat = np.sum(raster_mats, axis=0)

    # Create index array for the pre-stimulus time and the stimulation
    pre_stim_idx = np.logical_and(time_vec < 0, time_vec >= -pre_stim_window)
    stim_idx = np.logical_and(time_vec >=0, time_vec <= stim_len)

    # Get the stimulus start index
    stim_start = np.where(abs(time_vec) - 0 < 0.00000001)[0]

    # Get the signal types right before stimulus onset
    pre_stim_type = raster_mat[:, pre_stim_idx]

    # Get the signal changes during stimulation
    chgs, chgs_idx = utils.get_signal_switches([raster_mats[0][:,stim_idx], 
                                                raster_mats[1][:,stim_idx],
                                                raster_mats[2][:,stim_idx]], 
                                                bin_width=1/sampling_rate, tolerance=0.1)
    # Empty list to collect the changes upon stimulation
    event_changes = []

    # Iterate over all trials
    for triali in range(raster_mat.shape[0]):
        # If there were changes ...
        if chgs.size > 0 and chgs_idx[chgs_idx[:,0] == triali, 1].size > 0:
            # Get the column indices for the changes in the current trial
            trial_chg_idx = chgs_idx[chgs_idx[:,0] == triali, 1]
            # Determine the index of the first change (might be multiple due to overlaps)
            first_chg_idx = trial_chg_idx == trial_chg_idx[0]
            # Get the changes for the current trial
            trial_changes = chgs[chgs_idx[:,0] == triali, 1]
            # Select the signal type(s) after the first switch
            st_type = trial_changes[first_chg_idx][0]
        else:
            # If no switch was found, use the type right after stimulus start
            st_type = raster_mat[triali, stim_start+1][0]
            # Split overlap indices to corresponding types
            if st_type == 4.0:
                st_type = np.array([1, 3])
            elif st_type == 5.0:
                st_type = np.array([2, 3])
        
        # Get the pre stimulus signal type for the current trial
        if pre_stim_type[triali, -1] != 0.0:
            # If the type is non-zero right before stimulus onset, take this type
            pre_type = pre_stim_type[triali, -1]
        else:
            # If the type is zero right before stimulus onset, check in the whole allowed range for non-zero elements
            pre_stim_non_zero = np.where(pre_stim_type[triali, :] > 0)[0]
            # If a non-zero element was found, take this as the pre-stim type
            if pre_stim_non_zero.size > 0:
                pre_type = pre_stim_type[triali, pre_stim_non_zero[-1]]
            # Otherwise take zero as the pre-stim type
            else:
                pre_type = pre_stim_type[triali, -1]

        # Split overlap indices to corresponding types
        if pre_type == 4.0:
            pre_type = np.array([1, 3])
        elif pre_type == 5.0:
            pre_type = np.array([2, 3])

        # Store the signal change (depending on potential overlaps)
        if st_type.size > 1 and pre_type.size > 1:
            event_changes.append((pre_type[0], st_type[0]))
            event_changes.append((pre_type[1], st_type[1]))
        elif st_type.size > 1:
            event_changes.append((pre_type, st_type[0]))
            event_changes.append((pre_type, st_type[0]))
        elif pre_type.size > 1:
            event_changes.append((pre_type[0], st_type))
            event_changes.append((pre_type[0], st_type))
        else:
            event_changes.append((pre_type, st_type))

    # Convert to numpy array
    event_changes = np.array(event_changes)

    # Get all possible event type changes
    poss_event_changes = np.unique(event_changes, axis=0)
    # Count the number of occurrances of each possible event type change
    event_change_count = [ np.sum(np.logical_and(event_changes[:,0] == i, event_changes[:,1] == j)) for i,j in poss_event_changes ]
    # Add four to the second column to distinguish the previous types from the later ones (needed for the plot)
    poss_event_changes[:, 1] += 4
    # Set a color order for the nodes and links in the sandkey diagram
    colors.extend(colors)
    col_sandkey_boxes = [ to_hex(col) for col in colors ]
    col_sandkey_links = [ f'rgba{(to_rgba(colors[int(i)], alpha))}' for i in poss_event_changes[:,0] ]

    return poss_event_changes, event_change_count, col_sandkey_boxes, col_sandkey_links


