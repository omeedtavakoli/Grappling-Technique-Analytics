# Dominance in Grappling: Predictive Analytics of Positional Control

## Objective
To analyze and predict which grappling techniques and positions are more likely to result in control and submission in martial arts, using data-driven methods.

## Data Source
This study utilizes the "Grappling Techniques" dataset from Kaggle, compiled by Luca Besso. It includes techniques from Brazilian Jiu-Jitsu, Judo, and Wrestling. The dataset is under the Apache 2.0 License and is accessible [here](https://www.kaggle.com/datasets/liiucbs/grappling-techniques).

## Data Analysis
The dataset was preprocessed for missing values, and an analysis was conducted to determine the distribution of techniques and positions. Our analysis incorporates advanced statistical techniques and comparative studies across various martial arts styles.

## Visualizations
We created bar charts to depict the frequency of dominant versus defensive positions. Additional visualizations compare the effectiveness of techniques across Brazilian Jiu-Jitsu, Judo, and Wrestling, highlighting style-specific trends.

## Machine Learning
A RandomForestClassifier is employed to predict the effectiveness of positions in grappling. The model's performance was evaluated using metrics such as accuracy, precision, recall, and cross-validation. Feature engineering was applied to enhance predictive capabilities.

## Trend and Forecasting Analysis
The predictions from our model, coupled with extended analysis, offer insights that may indicate future trends in martial arts techniques.

## How to Use
- Install Python and the required libraries (Pandas, Matplotlib, scikit-learn).
- Execute `analysis.py` for initial analysis and data visualization.
- Refer to `extended_analysis.py` for advanced analysis and visualizations.

## File Descriptions
- `analysis.py`: Contains initial analysis, machine learning model training, and advanced statistical analysis with visualizations.
- `dataset.csv`: The dataset file used for the project (not included in this repository).

## Results
Detailed are the model's accuracy, precision, recall, and a classification report. Feature importance is evaluated to understand influential factors better. Key findings from our extended analysis are also presented.

## Contributing
Contributions are welcome. Fork the repository, commit your changes, and create a pull request. See our contribution guidelines for more details.

## License
This project uses the "Grappling Techniques" dataset under the Apache 2.0 License, as provided by Luca Besso on Kaggle.

## Contact
For inquiries or collaboration, connect with [Omeed Tavakoli](https://www.linkedin.com/in/omeed-tavakoli-b38685194/) on LinkedIn.

## Acknowledgments
Special thanks to the martial arts community, Kaggle, and Luca Besso for providing the dataset.
