"""
martial arts technique classifier
"""
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
import joblib

logging.basicConfig(                          # setup logging
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("martial_arts_classifier.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

CONFIG = {                                    # config params
    'dataset_path': '/path/to/your/dataset.csv',
    'output_dir': 'output',
    'random_state': 42,
    'test_size': 0.2,
    'cv_folds': 5,
    'model_params': {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
}

if not os.path.exists(CONFIG['output_dir']):  # create output dir
    os.makedirs(CONFIG['output_dir'])


def load_data(file_path):
    """
    load dataset from specified path with error handling
    """
    try:
        logger.info(f"Loading data from {file_path}")            # log start
        data = pd.read_csv(file_path)                            # read csv
        logger.info(f"Successfully loaded data with shape {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")            # log error
        return None


def explore_data(data, output_dir):
    """
    perform exploratory data analysis and generate visualizations
    """
    try:
        logger.info("Starting exploratory data analysis")         # log start
        
        logger.info(f"Dataset shape: {data.shape}")               # basic info
        logger.info("First five rows of the dataset:")
        logger.info(data.head())
        
        missing_values = data.isnull().sum()                      # check nulls
        logger.info(f"Missing values per column:\n{missing_values}")
        
        plt.figure(figsize=(10, 6))                               # null heatmap
        sns.heatmap(data.isnull(), yticklabels=False, cbar=False, cmap='viridis')
        plt.title('Missing Value Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'missing_values_heatmap.png'))
        plt.close()
        
        logger.info("Statistical summary:")                       # summary stats
        logger.info(data.describe(include='all'))
        
        for col in data.select_dtypes(include=['object']).columns:  # cat cols viz
            unique_vals = data[col].unique()
            logger.info(f"Unique values in '{col}': {unique_vals}")
            
            plt.figure(figsize=(12, 6))                           # bar charts
            counts = data[col].value_counts()
            sns.barplot(x=counts.index, y=counts.values)
            plt.title(f'{col} Counts')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{col}_counts.png'))
            plt.close()
        
        numerical_data = data.select_dtypes(include=['int64', 'float64'])  # num cols
        if not numerical_data.empty:                              # corr matrix
            plt.figure(figsize=(12, 10))
            correlation = numerical_data.corr()
            mask = np.triu(np.ones_like(correlation, dtype=bool))
            sns.heatmap(correlation, mask=mask, annot=True, cmap='coolwarm', linewidths=0.5)
            plt.title('Correlation Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
            plt.close()
        
        if 'Origin' in data.columns and 'Type' in data.columns:   # tech by style
            for style in data['Origin'].unique():
                subset = data[data['Origin'] == style]
                technique_counts = subset['Type'].value_counts()
                
                plt.figure(figsize=(12, 6))                       # dist by style
                sns.barplot(x=technique_counts.index, y=technique_counts.values)
                plt.title(f'Technique Distribution in Style: {style}')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'technique_dist_style_{style}.png'))
                plt.close()
        
        if len(numerical_data.columns) > 1:                       # pairplot
            sample_cols = list(numerical_data.columns)[:min(5, len(numerical_data.columns))]
            plt.figure(figsize=(15, 15))
            sns.pairplot(data[sample_cols])
            plt.savefig(os.path.join(output_dir, 'pair_plot.png'))
            plt.close()
            
        logger.info("Exploratory data analysis completed")        # log done
    
    except Exception as e:
        logger.error(f"Error during data exploration: {str(e)}")  # log error


def preprocess_data(data):
    """
    preprocess data, handle cat vars and missing vals
    """
    try:
        logger.info("Starting data preprocessing")                # log start
        
        if 'Position' not in data.columns:                        # check target
            logger.error("Target column 'Position' not found in the dataset")
            return None, None, None
            
        X = data.drop(['Position'], axis=1)                       # set features
        y = data['Position']                                      # set target
        
        if 'Name' in X.columns:                                   # drop id col
            X = X.drop(['Name'], axis=1)
            
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()  # find cat cols
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()  # find num cols
        
        logger.info(f"Categorical columns: {categorical_cols}")   # log cols
        logger.info(f"Numerical columns: {numerical_cols}")
        
        numerical_transformer = Pipeline(steps=[                  # num pipeline
            ('imputer', SimpleImputer(strategy='median')),        # fill nulls
            ('scaler', StandardScaler())                          # scale nums
        ])
        
        categorical_transformer = Pipeline(steps=[                # cat pipeline
            ('imputer', SimpleImputer(strategy='most_frequent')), # fill nulls
            ('onehot', OneHotEncoder(handle_unknown='ignore'))    # encode cats
        ])
        
        preprocessor = ColumnTransformer(                         # combine pipelines
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ]
        )
        
        logger.info("Data preprocessing completed")               # log done
        return X, y, preprocessor
        
    except Exception as e:
        logger.error(f"Error during data preprocessing: {str(e)}")  # log error
        return None, None, None


def train_evaluate_model(X, y, preprocessor, config, output_dir):
    """
    train, evaluate and save model
    """
    try:
        logger.info("Starting model training and evaluation")     # log start
        
        model_pipeline = Pipeline(steps=[                         # create pipeline
            ('preprocessor', preprocessor),                       # preprocess data
            ('classifier', RandomForestClassifier(random_state=config['random_state']))  # init model
        ])
        
        X_train, X_test, y_train, y_test = train_test_split(      # split data
            X, y, test_size=config['test_size'], 
            random_state=config['random_state'], stratify=y       # strat split
        )
        
        logger.info(f"Training set size: {X_train.shape}, Test set size: {X_test.shape}")  # log splits
        
        cv = StratifiedKFold(n_splits=config['cv_folds'], shuffle=True, random_state=config['random_state'])  # cv setup
        
        logger.info("Starting hyperparameter tuning with GridSearchCV")  # start tuning
        grid_search = GridSearchCV(                               # search setup
            model_pipeline, 
            param_grid={'classifier__' + key: value for key, value in config['model_params'].items()},
            cv=cv, 
            scoring='accuracy', 
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)                         # run search
        
        logger.info(f"Best parameters: {grid_search.best_params_}")  # log best params
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")  # log best score
        
        best_model = grid_search.best_estimator_                  # get best model
        
        model_path = os.path.join(output_dir, 'martial_arts_classifier.joblib')  # save path
        joblib.dump(best_model, model_path)                       # save model
        logger.info(f"Model saved to {model_path}")               # log save
        
        y_pred = best_model.predict(X_test)                       # get preds
        accuracy = accuracy_score(y_test, y_pred)                 # calc acc
        logger.info(f"Test set accuracy: {accuracy:.4f}")         # log acc
        
        clf_report = classification_report(y_test, y_pred, output_dict=True)  # gen report
        pd.DataFrame(clf_report).transpose().to_csv(              # save report
            os.path.join(output_dir, 'classification_report.csv')
        )
        
        plt.figure(figsize=(10, 8))                               # conf matrix
        cm = confusion_matrix(y_test, y_pred)                     # get matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=best_model.classes_, 
                   yticklabels=best_model.classes_)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))  # save viz
        plt.close()
        
        classifier = best_model.named_steps['classifier']         # get model
        
        if hasattr(classifier, 'feature_importances_'):           # if has importances
            preprocessor = best_model.named_steps['preprocessor'] # get preproc
            feature_names = []                                    # init names list
            
            for name, transformer, cols in preprocessor.transformers_:  # get feat names
                if name == 'cat':                                 # for cat cols
                    onehot = transformer.named_steps['onehot']    # get encoder
                    onehot.fit(X[cols])                           # fit to get cats
                    for i, col in enumerate(cols):                # for each col
                        feature_names.extend([f"{col}_{c}" for c in onehot.categories_[i]])  # add ohe names
                else:                                             # for num cols
                    feature_names.extend(cols)                    # add orig names
            
            importances = classifier.feature_importances_         # get importances
            
            if len(feature_names) == len(importances):            # if sizes match
                feature_importance_df = pd.DataFrame({            # create df
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values(by='Importance', ascending=False)  # sort by imp
                
                feature_importance_df.to_csv(os.path.join(output_dir, 'feature_importances.csv'), index=False)  # save csv
                
                plt.figure(figsize=(12, 8))                       # imp viz
                sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(20))  # top 20
                plt.title('Top 20 Feature Importances')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'feature_importances.png'))  # save viz
                plt.close()
                
                selector = SelectFromModel(classifier, threshold='median')  # feat select
                selector.fit(preprocessor.transform(X_train), y_train)  # fit selector
                
                selected_mask = selector.get_support()            # get selection
                selected_features = [feature for feature, selected in zip(feature_names, selected_mask) if selected]  # get names
                with open(os.path.join(output_dir, 'selected_features.txt'), 'w') as f:  # save list
                    for feature in selected_features:
                        f.write(f"{feature}\n")
                
                logger.info(f"Number of selected features: {len(selected_features)} out of {len(feature_names)}")  # log count
        
        logger.info("Model evaluation completed")                 # log done
        return best_model
        
    except Exception as e:
        logger.error(f"Error during model training and evaluation: {str(e)}")  # log error
        return None


def main():
    """
    main function to orchestrate process
    """
    try:
        logger.info("Starting martial arts technique classification pipeline")  # log start
        
        data = load_data(CONFIG['dataset_path'])                  # load data
        if data is None:                                          # check load
            logger.error("Failed to load data. Exiting.")
            return
        
        explore_data(data, CONFIG['output_dir'])                  # explore
        
        X, y, preprocessor = preprocess_data(data)                # preprocess
        if X is None or y is None:                                # check prep
            logger.error("Failed to preprocess data. Exiting.")
            return
        
        model = train_evaluate_model(X, y, preprocessor, CONFIG, CONFIG['output_dir'])  # train
        if model is None:                                         # check model
            logger.error("Failed to train and evaluate model. Exiting.")
            return
        
        logger.info("Martial arts technique classification pipeline completed successfully")  # log done
        
    except Exception as e:
        logger.error(f"Unexpected error in main function: {str(e)}")  # log error


if __name__ == "__main__":
    main()                                                        # run pipeline
