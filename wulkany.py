#%%
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

class VolcanoClassifier:
    def __init__(self, filepath, target_column):
        self.filepath = filepath
        self.target_column = target_column
        self.data = None
        self.label_encoders = {}
        self.scaler = None
        self.model = None

    def load_data(self):
        self.data = pd.read_csv(self.filepath)
        print("Data loaded successfully.")

    def preprocess_data(self):
        data = self.data

        print("Initial Data Info:\n", data.info())
        print("Summary Statistics:\n", data.describe())

        # Remove unnecessary columns
        columns_to_drop = ['Number', 'Name', 'Country']
        data = data.drop(columns=[col for col in columns_to_drop if col in data.columns], errors='ignore')

        # Remove rows where 'Last Known Eruption' is 'Unknown'
        data = data[data['Last Known Eruption'] != 'Unknown']

        # missing values
        for column in data.select_dtypes(include=['float64', 'int64']).columns:
            data[column] = data[column].fillna(data[column].mean())

        for column in data.select_dtypes(include=['object']).columns:
            data[column] = data[column].fillna("Unknown")

        # Visualizations
        print("Creating visualizations...")
        plt.figure(figsize=(20, 10))
        sns.countplot(data=data, x='Type')
        plt.xticks(rotation=90)
        plt.title("Distribution of Volcano Types")
        plt.show()

        plt.figure(figsize=(12, 6))
        sns.countplot(data=data, x='Region', hue='Region')
        plt.xticks(rotation=90)
        plt.title("Distribution of Volcano Types by Region")
        plt.show()

        # Normalize numeric data
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
        self.scaler = StandardScaler()
        data[numeric_cols] = self.scaler.fit_transform(data[numeric_cols])

        # Boxs for each numeric column
        plt.figure(figsize=(15, 12))
        for i, col in enumerate(numeric_cols):
            plt.subplot(3, 2, i+1)
            sns.boxplot(data=data, x='Type', y=col)
            plt.title(f"{col} by Volcano Type")
            plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

        # Correlation matrix
        numeric_data = data.select_dtypes(include=['float64', 'int64'])
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Matrix")
        plt.show()

        # Encode categorical features
        for column in data.select_dtypes(include=['object']).columns:
            if column == self.target_column:
                continue
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column])
            self.label_encoders[column] = le

        # Encode target column
        target_encoder = LabelEncoder()
        data[self.target_column] = target_encoder.fit_transform(data[self.target_column])
        self.label_encoders[self.target_column] = target_encoder

        self.data = data

    def train_model(self):
        data = self.data
        min_samples = 5
        data = data[data[self.target_column].isin(
            data[self.target_column].value_counts()[data[self.target_column].value_counts() >= min_samples].index
        )]

        # Split data
        X = data.drop(columns=[self.target_column])
        y = data[self.target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Oversampling
        smote = SMOTE(random_state=42, k_neighbors=2)
        X_train, y_train = smote.fit_resample(X_train, y_train)

        # Train model
        clf = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
        grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=3, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        # Evaluate
        self.model = grid_search.best_estimator_
        y_pred = self.model.predict(X_test)

        print("Best Parameters:", grid_search.best_params_)
        print("Classification Report:\n", classification_report(y_test, y_pred))
        print("Accuracy Score:", accuracy_score(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    def run(self):
        print("Starting process...")
        self.load_data()
        self.preprocess_data()
        self.train_model()
        print("Process completed.")

# Main Execution
if __name__ == "__main__":
    filepath = "data/database.csv"  
    target_column = "Region"  

    classifier = VolcanoClassifier(filepath, target_column)
    classifier.run()

#%%
