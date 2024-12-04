import pandas as pd
import plotly.express as px

def visualize_predictions():
    """
    This function takes in the predicted tree loss from the `prediction_InPrimaryForest.csv' file and then combines it with the
    `TreeCoverLoss_2001-2020_inPrimaryForest.csv` to show the tree loss in the past in addition to our predicted tree loss.
    """

    given = pd.read_csv('input_data/TreeCoverLoss_2001-2020_ByRegion.csv')
    predict = pd.read_csv('output_data/prediction_ByRegion.csv')

    gsum = given.groupby('Year')['TreeCoverLoss_ha'].sum().reset_index()
    psum = predict.groupby('Year')['TreeCoverLoss_ha'].sum().reset_index()

    gsum['Origin'] = 'Recorded'
    psum['Origin'] = 'Predicted'

    all_data = pd.concat([gsum,psum])

    fig = px.line(all_data,
                  x='Year',
                  y= 'TreeCoverLoss_ha',
                  color = 'Origin',
                  title = 'Global Tree Loss with Predicted Loss from 2020-2030',
                  labels = {'TreeCoverLoss_ha': 'Tree Loss (Ha)', 'Year': 'Year'},
                  markers = True)
    
    fig.show()
