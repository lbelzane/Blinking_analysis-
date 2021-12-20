import CD 
import Timetrace
import Intensity_histogram
from Intensity_histogram import ONLevel_func
import lifetime


data_directory = '/Users/lucienbelzane/Desktop/Columbia/2021-11-09/em1/'


# Parameters of the mesurement:
n               =       0.01          # Bin time (in s)
T_ac            =       600           # Acquisition time (in s)

# =============================================================================
# Timetrace
# =============================================================================

Plot_timetrace =        False         # Do you want to plot the timetrace ? 
Plot_zoom      =        False         # Do you want to plot a zoom of the timetrace ?

xmin           =        300           # Minimum time for the zoom 
xmax           =        302           # Maximum time fot the zoom 

Timetrace.timetrace(Plot_timetrace, Plot_zoom, xmin, xmax, data_directory)

# =============================================================================
# Intensity histogram
# =============================================================================

Plot_Int_hist     =     False         # Do you want to plot the intensity histogram ? 
Filter_histogram  =     False         # Do you want to filter the histogram ? 

Intensity_histogram.Int_hist(Plot_Int_hist, Filter_histogram, data_directory)

# =============================================================================
# Cumulative distribution
# =============================================================================

OFF_analysis    =       False         # Do you want the cumulative distribution (CD) of OFF periods ?
ON_analysis     =       False         # Do you want the cumulative distribution of OFF periods ?
ON_OFF_comp     =       False         # Do you want the comparison beetween CD OFF and CD ON ? 
model_power     =       False         # Do you want to fit the CD with a power law ? 
model_power_exp =       True          # Do you want to fit the CD with a power + exp law ?

OFFLevel        =       60            # OFF Level thresold.
ONLevel         =       150           # ON  Level thresold.
#Percent         =       0
#ONLevel = ONLevel_func(Percent, data_directory)

CD.CD(OFF_analysis,ON_analysis, ON_OFF_comp, n, OFFLevel, ONLevel, T_ac , model_power, model_power_exp, data_directory)

# =============================================================================
# Lifetime Analysis
# =============================================================================
Lifetime        =       True 
Dark_state      =       True          # Do you want lifetime decay for Dark state ?
Grey_state      =       True          # Do you want lifetime decay for Bright state ?
Bright_state    =       True          # Do you want lifetime decay for Grey state ? 

#OFFLevel        =       40
Percent         =       125
#ONLevel         =       ONLevel_func(Percent, data_directory)

#lifetime.Lifetime(OFF_state, ON_state,Grey_state, OFFLevel, ONLevel)

Create_FLID     =       False          # Do you want to create FLID images ? 
          # Do you want to plot the lifetime decay ? 

lifetime.create_lifetime_FLID(Create_FLID, Lifetime, Bright_state, Dark_state, Grey_state, OFFLevel, ONLevel, n, data_directory)