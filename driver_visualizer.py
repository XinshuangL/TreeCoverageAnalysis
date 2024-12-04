import pandas as pd
import plotly.express as px

def visualize_drivers(file_name):
    """
    This function takes in a .csv file of the dominant drivers of deforestation
    and then plots them compared to their tree cover loss and CO2 emission figures using plotly.
    """
    #check to see if a valid .csv file was input
    assert isinstance(file_name,str)
    assert len(file_name) >= 5
    assert file_name[-4:] == ".csv"

    # read the .csv file using pandas data frame.
    file = pd.read_csv(file_name)

    # create a line graph with each main driver with year vs tree loss in hectares
    fig = px.line(file,
        x = 'Year',
        y = 'TreeCoverLoss_ha',
        color = 'DriverType',
        title = "Tree Cover Loss by Dominant Deforestation Drivers from 2001 to 2020",
        labels = {'Year':'Year', 'TreeCoverLoss_ha':'Tree Cover Loss (Ha)', 'DriverType': 'Driver Type' },
        markers=True
    )

    fig.show()

    # create a line graph with each main driver with year vs CO2 emissions in Mega grams
    fig2 = px.line(file,
                   x = 'Year',
                   y = 'GrossEmissions_Co2_all_gases_Mg',
                   color = 'DriverType',
                   title = "Gross Emissions of CO2 by Dominant Drivers",
                   labels = {'Year':'Year', 'GrossEmissions_Co2_all_gases_Mg':'Gross Emissions of CO2 (Mg)', 'DriverType':'Driver Type'},
                   markers=True
                   )
    
    fig2.show()
