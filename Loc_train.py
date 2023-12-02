import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import top_k_accuracy_score
from sklearn import metrics 
#----------------------------------1. Load File------------------------------------
file = "D:/ML_exercise/Label_with_data.csv"
species = ['asparagus', 'bambooshoots', 'betel', 'broccoli', 'cauliflower', 'chinesecabbage', 'chinesechives', 'custardapple', 'grape', 'greenhouse', 'greenonion', 'kale', 'lemon', 'lettuce', 'litchi', 'longan', 'loofah', 'mango', 'onion', 'others', 'papaya', 'passionfruit', 'pear', 'pennisetum', 'redbeans', 'roseapple', 'sesbania', 'soybeans', 'sunhemp', 'sweetpotato', 'taro', 'tea', 'waterbamboo']
TOP_K_ERROR = 10

# load dataset
Data = pd.read_csv(file)
Data.head()

#split dataset in features and target variable
feature_cols = ['town_x', 'town_y']
X = Data[feature_cols] # Features
y = Data['Species'] # Target variable

#----------------------------------2. Data Prepocessing------------------------------
# Use train_test_split to split X y list
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

#----------------------------------3. ML Classify------------------------------------
# Random Forest
clf = RandomForestClassifier(n_estimators=33, criterion="entropy")

# Train
clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)
y_pred_k = clf.predict_proba(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("TOP {}:".format(TOP_K_ERROR),top_k_accuracy_score(y_test, y_pred_k, k=TOP_K_ERROR))
    
Statistical_dict = {"times": [0]*33,
                    "label": species}
Statistical_df = pd.DataFrame(Statistical_dict)

for i in range(len(y_pred_k)):
    output_dict = {"pred": y_pred_k[i],
                    "label": species}
    Out_dataframe = pd.DataFrame(output_dict)
    K_test = Out_dataframe.nlargest(TOP_K_ERROR, 'pred', keep='all')
    index = []
    for j in range(len(species)):
        index1 = K_test.index[(K_test["label"] == species[j])]
        index_value = index1.to_list()
        try:
            if index_value != []:
                index.append(index_value[0])
        except:
            continue
    for l in range(TOP_K_ERROR):        
        Statistical_df.at[index[l], "times"] += 1

Top_10_species = Statistical_df.nlargest(TOP_K_ERROR, 'times', keep='all')

Final_label_index = []
for i in range(len(species)):
    index2 = Top_10_species.index[Top_10_species["label"] == species[i]]
    index_value = index2.to_list()
    try:
        if index_value != []:
            Final_label_index.append(index_value[0])
    except:
        continue
print(Final_label_index)



