# Data Analysis: World of Warcraft Battleground Win Prediction

## Project Description
This project focuses on analyzing and predicting the outcomes and durations of the *Battleground* game mode in **World of Warcraft**.
The data was sourced from a private server and includes player statistics and match results.
The goal was to use statistical methods and machine learning to answer specific questions and build predictive models.

## Objectives
1. **Analytical Goals:**
   - Answer questions about win probabilities for specific classes, races, the impact of healers, and player activity during weekends.
2. **Predictive Goals:**
   - Predict:
     - The outcome of a Battleground (*win/loss*).
     - The duration of a Battleground.
     - Individual player performance.

## Dataset
The dataset was sourced from Kaggle: [Massive World of Warcraft PvP Dataset](https://www.kaggle.com/datasets/sosperec/massive-world-of-warcraft-pvp-mists-of-pandaria?select=games.csv).  
It includes:
- **586,149 matches**, with **195,838 Battlegrounds**.
- Player-level data (5,915,447 records), including race, class, kills, deaths, damage dealt/received, healing done/received, and match outcomes.

## Research Questions and Hypotheses
1. Are classes like *Paladin*, *Hunter*, *Warrior*, and *Death Knight* better than others in terms of win probability?
2. Is the *Human* race superior to others in terms of win probability?
3. Does having more healers increase the chance of winning?
4. Are players more active during weekends?

## Methodology

### **1. Data Preprocessing**
- **Filtering:** Removed inconsistent data (e.g., arena matches mislabeled as Battlegrounds).
- **Transformation:** 
  - Converted time fields into seconds since epoch.
  - Applied *One-Hot Encoding* for categorical attributes (race, class, map, etc.).
- **Normalization:** Scaled numerical values for compatibility with machine learning models.

### **2. Statistical Analysis**
- Correlation analysis between game statistics.
- PCA (Principal Component Analysis) for dimensionality reduction.
- Computation of averages and standard deviations to answer research questions.

### **3. Machine Learning**
Two approaches were implemented:
1. **Submodel Approach:** Separate models for each task (2 regressors + 1 classifier).
2. **End-to-End Model:** A single neural network solving all tasks simultaneously.

Technologies used:
- Python library `torch` for neural networks.
- Scripts: `machine_learning.py`, `dataloader.py`.

## Results

### Statistical Analysis
1. Classes like *Priest* and *Monk* had higher win rates compared to classes like *Rogue*.
2. The *Human* race showed the highest win probability, confirming the hypothesis.
3. A higher number of healers positively influenced win rates.
4. Player activity was indeed higher on weekends.

### Machine Learning
- Models successfully predicted outcomes with satisfactory accuracy.
- The End-to-End approach outperformed submodels due to its ability to learn relationships between inputs automatically.

## How to Run the Project

1. **Download Data:**
   Run `download.py` to fetch the dataset.

2. **Preprocess Data:**
   Execute `dataloader.py` to prepare the dataset for analysis.

3. **Train Models:**
   Run `machine_learning.py` to train neural networks.

4. **View Results:**
   Outputs will be saved as graphs or printed to the console.
