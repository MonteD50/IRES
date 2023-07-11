import pandas as pd
import numpy as np
from somethingelse import ClassAssRules
from sklearn.metrics import accuracy_score


attribute_names = [
  "class_name", "handicapped-infants", "water-project-cost-sharing",
  "adoption-of-the-budget-resolution", "physician-fee-freeze",
  "el-salvador-aid", "religious-groups-in-schools", "anti-satellite-test-ban",
  "aid-to-nicaraguan-contras", "mx-missile", "immigration",
  "synfuels-corporation-cutback", "education-spending",
  "superfund-right-to-sue", "crime", "duty-free-exports",
  "export-administration-act-south-africa"
]

#attribute_names = [x.replace('-', '_') for x in attribute_names]

#df2 = pd.read_csv("house-votes-84.data", names=attribute_names)


df2 = pd.read_csv("Nursary.csv")

# Replace '?' with NaN
df2.replace('?', np.nan, inplace=True)

df2.dropna(inplace=True)

# Reset the index
df2.reset_index(drop=True, inplace=True)

# Drop the class_name column
df_without_class = df2.drop('class_name', axis=1, inplace=False)
df_encoded = pd.get_dummies(df_without_class)


target_class = df2['class_name'].tolist()

print(df2.shape)

#classAssRules = ClassAssRules(df2, df_encoded, target_class, 'class_name', get_top_coverage = 3, graph = False)
#classAssRules.run()

#prediction = classAssRules.predict(df_encoded)

# Calculate accuracy
#pred = prediction['prediction'].tolist()
#accuracy = accuracy_score(target_class, pred)
#print("Accuracy:", accuracy)

def calculate_accuracy(actual: list, predicted_df: pd.DataFrame):
    count = 0
    correct = 0
    for index, row in predicted_df.iterrows():
        predicted_class = row['prediction']
        actual_class = actual[count]
        if predicted_class == actual_class:
          correct += 1
        # If rule doesnt cover (ie. everything is 0), mark as correct
        elif row['prediction_count'] == 0:
          correct += 1
        else:
          # If prediction_count is the same for all classes, mark as correct
          is_same = True
          for col in predicted_df.columns:
              if col.startswith('prediction_count_for_'):
                  if row[col] != row['prediction_count']:
                      is_same = False 
          if is_same:
              print("Is same:", row['prediction_count'], row['prediction_count_for_spec_prior'], row['prediction_count_for_priority'], row['prediction_count_for_not_recom'])
              correct += 1

        count += 1  
        
    return correct / len(actual)


# 10-fold CV
def cross_validation(data_og, data_encoded, labels, num_folds=10):
    fold_size = len(data_encoded) // num_folds
    accuracy_scores = []

    for fold in range(num_folds):
        validation_data_encoded = data_encoded.iloc[fold*fold_size:(fold+1)*fold_size]

        train_data_encoded = pd.concat([data_encoded.iloc[:fold*fold_size], data_encoded.iloc[(fold+1)*fold_size:]])
        #train_labels = pd.concat([labels.iloc[:fold*fold_size], labels.iloc[(fold+1)*fold_size:]])
        train_data_og = pd.concat([data_og.iloc[:fold*fold_size], data_og.iloc[(fold+1)*fold_size:]])
        
        # Split the data into training and validation sets
        #validation_data_encoded = data_encoded[fold*fold_size:(fold+1)*fold_size]
        validation_labels = labels[fold*fold_size:(fold+1)*fold_size]
        #train_data_encoded = np.concatenate([data_encoded[:fold*fold_size], data_encoded[(fold+1)*fold_size:]])
        train_labels = np.concatenate([labels[:fold*fold_size], labels[(fold+1)*fold_size:]])
        #train_data_og = np.concatenate([data_og[:fold*fold_size], data_og[(fold+1)*fold_size:]])

        classAssRules = ClassAssRules(train_data_og, train_data_encoded, 
                                      train_labels, 'class_name', 
                                      min_support = 0.01, 
                                      min_confidence = 0.6, 
                                      max_rules = 1000, 
                                      get_top_coverage = 3, 
                                      num_clusters = {'spec_prior': 13, 'priority': 13, 'not_recom': 14, 'very_recom': 2, 'recommend': 1},
                                      graph = False,
                                      method = 'kmodes')
        classAssRules.run()

        prediction = classAssRules.predict(validation_data_encoded)

        # Calculate accuracy
        pred = prediction['prediction'].tolist()


        accuracy = calculate_accuracy(validation_labels, prediction)
        accuracy_scores.append(accuracy)
        print("Accuracy:", accuracy, "for fold", fold)

    mean_accuracy = np.mean(accuracy_scores)
    sd_accuracy = np.std(accuracy_score)
    print("CV Accuracy:", mean_accuracy, "CV Std:", sd_accuracy)

cross_validation(df2, df_encoded, target_class)

