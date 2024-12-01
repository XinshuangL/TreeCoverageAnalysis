import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

class TreeLossVisualizer:
    def __init__(self, tree_loss_file, country_code_file='input_data/country_code_info.csv'):
        """
        Initialize the TreeLossVisualizer with data files.
        
        :param tree_loss_file: Path to the tree loss CSV file.
        :param country_code_file: Path to the country code information CSV file.
        """
        # Load data
        self.tree_loss_df = pd.read_csv(tree_loss_file)
        self.country_code_info_df = pd.read_csv(country_code_file)

        # Merge datasets
        self.merged_df = self.tree_loss_df.merge(
            self.country_code_info_df[['alpha-3', 'region', 'sub-region']],
            left_on='CountryCode',
            right_on='alpha-3',
            how='inner'
        )

    def generate_figure(self, fig, output_mode, file_name=None):
        """
        Display or save a Plotly figure based on the output mode.
        
        :param fig: The Plotly figure to handle.
        :param output_mode: Either 'show' to display or 'save' to save as HTML.
        :param file_name: The name of the HTML file if output_mode is 'save'.
        """
        if output_mode == 'show':
            fig.show()
        elif output_mode == 'save' and file_name:
            fig.write_html(file_name)
        else:
            raise ValueError("Invalid output_mode. Choose 'show' or 'save', and provide file_name for 'save'.")

    def map_country(self, data_column, title, color_scale, output_mode='show', file_name=None):
        """
        Map visualization of countries by value (tree loss or emissions).
        """
        fig = px.choropleth(
            self.merged_df,
            locations="CountryCode",
            locationmode="ISO-3",
            color=data_column,
            animation_frame="Year",
            title=title,
            color_continuous_scale=color_scale,
            labels={data_column: data_column.replace("_", " ").title()},
            hover_name="region"
        )

        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1000

        self.generate_figure(fig, output_mode, file_name)

    def map_region(self, data_column, title, color_scale, output_mode='show', file_name=None):
        """
        Map visualization of regions by aggregated value (tree loss or emissions).
        """
        subregion_year_loss = self.merged_df.groupby(['Year', 'sub-region'])[data_column].sum().reset_index()
        merged_df_with_subregion_loss = self.merged_df.merge(
            subregion_year_loss,
            on=['Year', 'sub-region'],
            suffixes=('', '_total') 
        )
        final_map_df = merged_df_with_subregion_loss.drop_duplicates(subset=["CountryCode", "Year"])

        fig = px.choropleth(
            final_map_df,
            locations="alpha-3",
            locationmode="ISO-3",
            color=f"{data_column}_total",
            animation_frame="Year",
            title=title,
            color_continuous_scale=color_scale,
            labels={f"{data_column}_total": data_column.replace("_", " ").title()}
        )

        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1000

        self.generate_figure(fig, output_mode, file_name)

    def top_countries(self, data_column, title, output_mode='show', file_name=None):
        """
        Bar chart of top 10 countries by value (tree loss or emissions).
        """
        top_10_countries_per_year = (
            self.merged_df.groupby("Year", group_keys=False)
            .apply(lambda x: x.nlargest(10, data_column))
            .reset_index()
        )

        fig = px.bar(
            top_10_countries_per_year,
            x=data_column,
            y="CountryCode",
            color="region",
            animation_frame="Year",
            title=title,
            labels={data_column: data_column.replace("_", " ").title(), "CountryCode": "Country Code"},
            orientation="h"
        )

        fig.update_layout(
            xaxis_title=data_column.replace("_", " ").title(),
            yaxis_title="Country Code",
            xaxis=dict(range = [0, 1.01*(top_10_countries_per_year[data_column].max())]),
            yaxis=dict(categoryorder="total ascending"),
            showlegend=True       
        )

        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1000  

        self.generate_figure(fig, output_mode, file_name)

    def top_regions(self, data_column, title, output_mode='show', file_name=None):
        """
        Bar chart of regions by aggregated value (tree loss or emissions).
        """
        subregion_year_loss = self.merged_df.groupby(['Year', 'sub-region'])[data_column].sum().reset_index()
        
        fig = px.bar(
            subregion_year_loss,
            x=data_column,
            y="sub-region",
            color="sub-region",
            animation_frame="Year",
            title=title,
            labels={data_column: data_column.replace("_", " ").title(), "sub-region": "Sub-Region"},
            orientation="h"
        )

        fig.update_layout(
            xaxis_title=data_column.replace("_", " ").title(),
            yaxis_title="Sub-Region",
            xaxis=dict(range = [0, 1.01*(subregion_year_loss[data_column].max())]),
            yaxis=dict(categoryorder="total ascending"),
            margin={'t':50, 'b':50},
            showlegend=False
        )

        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1000  

        self.generate_figure(fig, output_mode, file_name)

    def country_trend(self, country_code, data_column, title, line_color, output_mode='show', file_name=None):
        """
        Line chart for a specific country (tree loss or emissions).
        """
        country_data = self.merged_df[self.merged_df['CountryCode'] == country_code]

        fig = go.Figure(
            data=go.Scatter(
                x=country_data['Year'],
                y=country_data[data_column],
                mode='lines+markers',
                marker=dict(color=line_color),
                name=country_code
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title="Year",
            yaxis_title=data_column.replace("_", " ").title(),
            showlegend=False
        )
        self.generate_figure(fig, output_mode, file_name)

    def global_trend(self, data_column, title, line_color, output_mode='show', file_name=None):
        """
        Line chart for global trend (tree loss or emissions).
        """
        total_data_per_year = self.merged_df.groupby('Year')[data_column].sum().reset_index()

        fig = go.Figure(
            data=go.Scatter(
                x=total_data_per_year['Year'],
                y=total_data_per_year[data_column],
                mode='lines+markers',
                marker=dict(color=line_color),
                name=f'Total {data_column.replace("_", " ").title()}'
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title="Year",
            yaxis_title=data_column.replace("_", " ").title(),
            showlegend=False
        )
        self.generate_figure(fig, output_mode, file_name)

if __name__ == "__main__":
    visualizer = TreeLossVisualizer('./input_data/TreeCoverLoss_2001-2020_ByRegion.csv', 'input_data/country_code_info.csv')

    # Tree Loss saving
    visualizer.map_country("TreeCoverLoss_ha", "Global Tree Cover Loss by Country", "Reds", output_mode="save", file_name="output_data/html/tree_loss_map_country.html")
    visualizer.map_region("TreeCoverLoss_ha", "Global Tree Cover Loss by Sub-Region", "Reds", output_mode="save", file_name="output_data/html/tree_loss_map_region.html")
    visualizer.top_countries("TreeCoverLoss_ha", "Top 10 Countries by Tree Cover Loss", output_mode="save", file_name="output_data/html/tree_loss_top_countries.html")
    visualizer.top_regions("TreeCoverLoss_ha", "Tree Cover Loss by Sub-Region", output_mode="save", file_name="output_data/html/tree_loss_top_regions.html")
    visualizer.country_trend("USA", "TreeCoverLoss_ha", "Tree Cover Loss for USA by Year", "blue", output_mode="save", file_name="output_data/html/tree_loss_country_trend.html")
    visualizer.global_trend("TreeCoverLoss_ha", "Total Tree Cover Loss Worldwide by Year", "green", output_mode="save", file_name="output_data/html/tree_loss_global_trend.html")

    # Gross Emissions saving
    visualizer.map_country("GrossEmissions_Co2_all_gases_Mg", "Global Gross Emissions by Country", "Blues", output_mode="save", file_name="output_data/html/emissions_map_country.html")
    visualizer.map_region("GrossEmissions_Co2_all_gases_Mg", "Global Gross Emissions by Sub-Region", "Blues", output_mode="save", file_name="output_data/html/emissions_map_region.html")
    visualizer.top_countries("GrossEmissions_Co2_all_gases_Mg", "Top 10 Countries by Gross Emissions", output_mode="save", file_name="output_data/html/emissions_top_countries.html")
    visualizer.top_regions("GrossEmissions_Co2_all_gases_Mg", "Gross Emissions by Sub-Region", output_mode="save", file_name="output_data/html/emissions_top_regions.html")
    visualizer.country_trend("USA", "GrossEmissions_Co2_all_gases_Mg", "Gross Emissions for USA by Year", "orange", output_mode="save", file_name="output_data/html/emissions_country_trend.html")
    visualizer.global_trend("GrossEmissions_Co2_all_gases_Mg", "Total Gross Emissions Worldwide by Year", "purple", output_mode="save", file_name="output_data/html/emissions_global_trend.html")
