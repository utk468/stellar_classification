import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

df= pd.read_csv("star_classification.csv")
print(df.head())

rows,columns = df.shape
print(f"Number of rows/examples: {rows}")
print(f"Number of columns/features: {columns}")

print(df.info())

print(df.describe())

print(df.duplicated().sum())


# Calculate unique values for each column and sort them
unique_values = df.nunique().sort_values()
plt.figure(figsize=(10, 6))
unique_values.plot(kind='barh', color='darkblue')  
plt.title('Number of Unique Values per Column')
plt.xlabel('Number of Unique Values')
plt.ylabel('Columns')
plt.tight_layout()
plt.show()




class_counts = df['class'].value_counts()
colors = ['#00008B', '#FFA500', '#000000']
plt.bar(class_counts.index, class_counts.values, color=colors)
plt.xlabel('Class', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Distribution of Classes', fontsize=14)
plt.show()



class_counts = df['class'].value_counts()
colors = ['#00008B', '#FFA500', '#000000']
plt.figure(figsize=(7, 7))
plt.pie(class_counts.values, labels=class_counts.index, colors=colors, autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Classes', fontsize=14)
plt.show()


#remember it use in many probability density fiunctiion
# Drop rows where 'z' is less than 0
df = df[df['z'] >= 0]
features = ['u', 'g', 'r', 'i', 'z']
palette = ['#1e3a5f', '#AA336A', '#76ABDF']

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 18))
axes = axes.flatten()

for i, feature in enumerate(features):
    ax = axes[i]  
    for j, class_name in enumerate(df['class'].unique()):
        class_data = df[df['class'] == class_name][feature]
        
        
        sns.kdeplot(class_data, ax=ax, label=f'{class_name}', color=palette[j % len(palette)], fill=True, alpha=0.3)
    
    
    ax.set_title(f'Probability Density Function of {feature} by Class')
    ax.set_xlabel('Magnitude')
    ax.set_ylabel('Density')
    ax.legend(title='Class')


axes[-1].axis('off')
plt.subplots_adjust(hspace=0.333) 
plt.show()



#average of each class with respect to u ,g, r, i, z
average_values = df.groupby('class')[['u', 'g', 'r', 'i', 'z']].mean().reset_index()


average_values_melted = average_values.melt(id_vars='class', 
                                            value_vars=['u', 'g', 'r', 'i', 'z'], 
                                            var_name='Band', value_name='Average')


plt.figure(figsize=(10, 6))
sns.barplot(x='class', y='Average', hue='Band', data=average_values_melted, palette='Blues')
plt.title('Average of u, g, r, i, z for Each Class', fontsize=14)
plt.xlabel('Class', fontsize=12)
plt.ylabel('Average Value', fontsize=12)
plt.tight_layout()
plt.show()







#distribution of alpha with respect to classes
plt.figure(figsize=(10, 6))
for class_name in df['class'].unique():
    sns.histplot(df[df['class'] == class_name]['alpha'], label=f'Class {class_name}', kde=True, bins=30, alpha=0.6)


plt.title('Distribution of Alpha for Each Class', fontsize=14)
plt.xlabel('Alpha', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(title='Class')
plt.tight_layout()
plt.show()





#distribution of delta with respect to classes
plt.figure(figsize=(10, 6))

for class_name in df['class'].unique():
    sns.histplot(df[df['class'] == class_name]['delta'], label=f'Class {class_name}', kde=True, bins=30, alpha=0.6)

plt.title('Distribution of Delta for Each Class', fontsize=14)
plt.xlabel('Delta', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(title='Class')
plt.tight_layout()
plt.show()





#distribution of redshift with respect to classes
plt.figure(figsize=(10, 6))


for class_name in df['class'].unique():
    sns.histplot(df[df['class'] == class_name]['redshift'], label=f'Class {class_name}', kde=True, bins=30, alpha=0.6)

plt.title('Distribution of redshift for Each Class', fontsize=14)
plt.xlabel('redshift', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(title='Class')
plt.tight_layout()
plt.show()


#scatter plot between alpha and delta with respect of 3 classes 
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='alpha', y='delta', hue='class', palette='Set2', s=70, alpha=0.8)
plt.title('Scatter Plot of Alpha vs Delta by Class', fontsize=14)
plt.xlabel('Alpha', fontsize=12)
plt.ylabel('Delta', fontsize=12)
plt.legend(title='Class')
plt.tight_layout()
plt.show()



# Step 1: Encode the 'class' column (without modifying original df)
df_copy = df.copy()
label_encoder = LabelEncoder()
df_copy['class_encoded'] = label_encoder.fit_transform(df_copy['class'])
numeric_df = df_copy.select_dtypes(include=[int, float])
corr_matrix = numeric_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', linewidths=0.5, cbar=True)
plt.title('Correlation Heatmap (Excluding rerun_ID)', fontsize=16)
plt.tight_layout()
plt.show()







# Drop unnecessary columns
columns_to_drop = ['obj_ID', 'run_ID', 'rerun_ID', 'cam_col', 'field_ID', 'spec_obj_ID', 'fiber_ID','MJD']
numeric_df.drop(columns=columns_to_drop, inplace=True)
corr_matrix = numeric_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', linewidths=0.5, cbar=True)
plt.title('Correlation Heatmap', fontsize=16)
plt.tight_layout()
plt.show()


# finding outlier
def find_outliers_iqr(dataframe):
    outlier_indices = {}

    for column in dataframe.select_dtypes(include=[np.number]).columns:
        Q1 = dataframe[column].quantile(0.25)
        Q3 = dataframe[column].quantile(0.75)
        IQR = Q3 - Q1


        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR


        outlier_indices[column] = dataframe[(dataframe[column] < lower_bound) | (dataframe[column] > upper_bound)].index.tolist()

    return outlier_indices


outliers = find_outliers_iqr(df)


for column, indices in outliers.items():
    if indices:
        print(f"\nOutliers for column '{column}':")
        print(f"  Outlier indices: {indices}")
    else:
        print(f"\nNo outliers found for column '{column}'.")


print(df.shape)        


#graph with outlier
df['class_encoded'] = label_encoder.fit_transform(df_copy['class'])
sns.set(style="whitegrid")
melted_df = df.melt(var_name='Feature', value_name='Value', value_vars=df.select_dtypes(include=[np.number]).columns)
plt.figure(figsize=(12, 8))
sns.boxplot(x='Feature', y='Value', data=melted_df)
plt.title('Box Plots of Numeric Features')
plt.xticks(rotation=45)
plt.xlabel('Features')
plt.ylabel('Values')
plt.show()


scaler = RobustScaler()
df[['u_scaled', 'g_scaled', 'r_scaled', 'i_scaled', 'z_scaled']] = scaler.fit_transform(df[['u', 'g', 'r', 'i', 'z']])

# Drop unnecessary columns
columns_to_drop = ['obj_ID', 'run_ID', 'rerun_ID', 'cam_col', 'field_ID', 'spec_obj_ID', 'fiber_ID']
df.drop(columns=columns_to_drop, inplace=True)


label_encoder = LabelEncoder()
df['class_encoded'] = label_encoder.fit_transform(df['class'])


X = df[['alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'redshift']]  
y = df['class_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


pipeline_rf = Pipeline([
    ('scaler', RobustScaler()),       
    ('rf_classifier', RandomForestClassifier(class_weight='balanced', random_state=42) ) # Random Forest Classifier
])

pipeline_dt = Pipeline([
    ('scaler', RobustScaler()),        
    ('dt_classifier', DecisionTreeClassifier(class_weight='balanced',random_state=42))  
])

pipeline_xgb = Pipeline([
    ('scaler', RobustScaler()),        
    ('xgb_classifier', XGBClassifier(random_state=42))  
])

pipeline_svc = Pipeline([
    ('scaler', RobustScaler()),        
    ('svc_classifier', SVC(random_state=42))  
])


pipelines = [pipeline_rf, pipeline_dt, pipeline_xgb,pipeline_svc]
pipe_dict = {0: "RandomForest", 1: "DecisionTree", 2: "XGBoost",3:"SVC"}


cv_results = []
for i, pipe in enumerate(pipelines):
    cv_score = cross_val_score(pipe, X_train, y_train, scoring="accuracy", cv=10) 
    cv_results.append(cv_score)
    
    print(f"{pipe_dict[i]}: {cv_score.mean()} ± {cv_score.std()}")  



pipeline_rf.fit(X_train, y_train)

y_pred = pipeline_rf.predict(X_test)

report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
print(report)


with open('rf_model.pkl', 'wb') as model_file:
    pickle.dump(pipeline_rf, model_file)

with open('label_encoder.pkl', 'wb') as le_file:
    pickle.dump(label_encoder, le_file)


