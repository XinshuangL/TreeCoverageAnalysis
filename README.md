# **Tree Coverage Analysis**

## **Team 9**

### **Authors**
- Emma Wrightson  
- Xinshuang Liu  
- Haoyu Hu  
- Mazeyu Ji  
- Qi Cao  

---

## **Project Description**
This project investigates global tree loss across countries and regions from 2001 to 2020. It examines the gross CO2 emissions associated with tree loss and explores its primary causes. The project includes predictive modeling to forecast future tree loss trends and their potential correlation with CO2 emissions.

---

## **Code Overview**
### What the Code Does
1. **Data Analysis and Visualization**:
   - Analyzes historical data on tree loss and CO2 emissions.
   - Creates insightful visualizations to illustrate trends.
2. **Prediction Modeling**:
   - Develops machine learning models to predict future tree loss trends.
3. **Insight Generation**:
   - Correlates tree loss with CO2 emissions and identifies key contributing factors.

---

## **File Structure**

```bash
TreeCoverageAnalysis/
│
├── input_data/                         # Contains raw data files.
│   ├── country_code_info.csv           # Dataset with country codes.
│   ├── TreeCoverLoss_2001-2020_InPrimaryForest.csv  # Primary forest dataset.
│   ├── TreeCoverLoss_2001-2020_ByRegion.csv         # Main dataset by region.
│   └── TreeCoverLoss_2001-2020_DominantDrivers.csv  # Dataset on dominant drivers.
│
├── output_data/                        # Contains prediction results and visualizations.
│   ├── html/                           # Interactive visualizations for tree cover loss and CO2.
│   ├── prediction_ByRegion.csv         # Predictions based on the main dataset.
│   └── prediction_InPrimaryForest.csv  # Predictions based on the primary forest dataset.
│
├── Statistics/                         # Exploratory data analysis and statistics.
│   ├── Basic Statistics Countries.json         # Country-level statistics on tree cover loss and CO2.
│   ├── Basic Statistics Drivers.json           # Driver-level statistics on tree cover loss and CO2.
│   ├── Distribution_of_Records_Per_Country.png # Record counts by country.
│   ├── Distribution_of_Records_Per_Driver.png  # Record counts by driver type.
│   ├── Distribution_of_Records_Per_Year_Countries.png # Yearly record counts by country.
│   └── Distribution_of_Records_Per_Year_Drivers.png   # Yearly record counts by driver type.
│
├── dataset.py                        # Dataset class definition for loading and preprocessing data.
├── Gaussian_process_regression.ipynb # Jupyter notebook for prediction modeling using Gaussian process regression.
├── main.py                           # Main script to run the project pipeline.
├── prediction_baselines_statistical_methods.py # Script for baseline prediction models.
├── README.md                         # Documentation for the project.
├── visualization.ipynb               # Jupyter notebook for generating visualizations.
└── visualizer.py                     # Python script for visualization tasks.
```

---

## **How to Run the Code**
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/XinshuangL/TreeCoverageAnalysis.git
   cd TreeCoverageAnalysis
   ```
2. **Install Dependencies**:
    - Ensure you have Python 3.8+ installed.
    - Install the required libraries using the command:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the main script**:
   ```bash
   python3 main.py
   ```

---

## **Third Party Libraries**
In order to predict and visualize our data, we incorporated several third party libraries including:
- plotly
- pandas
- os
- matplotlib
- numpy
- torch
- json
- math
- warnings
- sklearn
- seaborn

---

## **Final Presentation**

---

## **Summary**
This repository offers a comprehensive study on tree coverage loss and its environmental impact, using data analysis, machine learning, and advanced visualization. Follow the instructions above to explore the code and reproduce the results.