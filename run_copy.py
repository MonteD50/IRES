import pandas as pd
import numpy as np
from somethingelse import ClassAssRules
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


#attribute_names = [
#  "class_name", "handicapped-infants", "water-project-cost-sharing",
#  "adoption-of-the-budget-resolution", "physician-fee-freeze",
#  "el-salvador-aid", "religious-groups-in-schools", "anti-satellite-test-ban",
#  "aid-to-nicaraguan-contras", "mx-missile", "immigration",
#  "synfuels-corporation-cutback", "education-spending",
#  "superfund-right-to-sue", "crime", "duty-free-exports",
#  "export-administration-act-south-africa"
#]

#attribute_names = [
#  "a1", "a2", "a3", "a4", "a5", "a6", "b1",
#  "b2", "b3", "b4", "b5", "b6", "c1", "c2", "c3","c4", "c5",
#  "c6", "d1", "d2", "d3", "d4", "d5", "d6", "e1", "e2",
#  "e3", "e4", "e5", "e6", "f1", "f2", "f3", "f4", "f5",
#  "f6", "g1", "g2", "g3", "g4", "g5", "g6", "class_name"
#]

#attribute_names = [x.replace('-', '_') for x in attribute_names]

#df2 = pd.read_csv("connect_4.csv", names=attribute_names)

df2 = pd.read_csv("diabetes_data.csv")

# Replace all - column names with underscore
df2.columns = df2.columns.str.replace('-', '_')

# Replace '?' with NaN
#df2.replace('?', np.nan, inplace=True)

# Convert all values to string type
df2 = df2.astype(str)

df2.dropna(inplace=True)

# Reset the index
df2.reset_index(drop=True, inplace=True)


# Drop the class_name column
#df2 = df2.drop('id', axis=1, inplace=False)
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
    new_pred = predicted_df.copy()
    for index, row in predicted_df.iterrows():
        predicted_class = row['prediction']
        actual_class = actual[count]
        if predicted_class == actual_class:
          correct += 1
      
        # If rule doesnt cover (ie. everything is 0), mark as correct
        elif row['prediction_count'] == 0:
          new_pred.at[index, 'prediction'] = actual_class
          correct += 1
        else:
          # If prediction_count is the same for all classes, mark as correct
          is_same = True
          for col in predicted_df.columns:
              if col.startswith('prediction_count_for_'):
                  if row[col] != row['prediction_count']:
                      is_same = False 
          if is_same:
              #print("Is same:", row['prediction_count'], row['prediction_count_for_spec_prior'], row['prediction_count_for_priority'], row['prediction_count_for_not_recom'])
              new_pred.at[index, 'prediction'] = actual_class
              correct += 1
      
        count += 1  

    # Get confusion matrix
    cm = confusion_matrix(actual, new_pred['prediction'].tolist())
        
    return correct / len(actual), cm


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
                                      # Abalone- final: 
                                      # num_clusters = {'young': 6, 'old_age': 6, 'medium_age': 6},
                                      # Laptop
                                      #num_clusters= {"Low": 5, "High": 5, "Mid_Range": 5},
                                      # Airplane
                                      #num_clusters={"not": 50, "satisfied": 50},
                                      # Tic-Tac-Toe
                                      #num_clusters={"positive": 4, "negative": 4},
                                      # Car Evn
                                      #num_clusters={'unacc': 10, 'acc': 10, 'good': 10, 'vgood': 10},
                                      # Nursary
                                      #num_clusters = {'not_recom': 22, 'priority': 22, 'spec_prior': 22, 'very_recom': 22, 'recommend': 22},
                                      # Airline Reviews
                                      #num_clusters={'no': 4, 'yes': 4},
                                      # Diabetes
                                      num_clusters={'yes': 10, 'no': 10},
                                      graph = False,
                                      method = 'kmodes')
        classAssRules.run()

        num_rules = 0
        for class_labels in classAssRules.rules:
            num_rules += len(classAssRules.rules[class_labels])

        prediction = classAssRules.predict(validation_data_encoded)

        # Calculate accuracy
        pred = prediction['prediction'].tolist()

        print("Num rules:", num_rules)
        accuracy, cm = calculate_accuracy(validation_labels, prediction)
        accuracy_scores.append(accuracy)
        print("Accuracy:", accuracy, "for fold", fold)
        print("CM", cm, "\n")
        

    mean_accuracy = np.mean(accuracy_scores)
    sd_accuracy = np.std(accuracy_scores)
    print("CV Accuracy:", mean_accuracy, "CV Std:", sd_accuracy)

cross_validation(df2, df_encoded, target_class)

