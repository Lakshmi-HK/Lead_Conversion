from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
import my_modules_model
import numpy as np

path = "../data/Leads.csv"

# load file
df = my_modules_model.load_df(path)
print(df.columns)

#preprocessing
my_modules_model.sanity_check(df)
my_modules_model.handle_missing_values(df)
my_modules_model.handle_categorical_cols(df)

x= df.drop("Converted",axis=1)
y= df['Converted']

print("Before handling outliers\n")
print(x.columns)

x = my_modules_model.outlier_handle(x)

# split of dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, stratify = y, random_state = 7)

model = LGBMClassifier()


# Train model
my_modules_model.Train_model(x_train, y_train, model)

# Evaluate model
predictions = model.predict(x_test)
my_modules_model.evalaute_model(y_test, predictions)

# Evaluate model
#threshold = 0.4 
#prediction_prob = model.predict_proba(x_test)[:,1]
#predictions = prediction_prob > threshold
#predictions = np.where(predictions==False,0,1)
#my_modules_model.evalaute_model(y_test, predictions)

# Save model
my_modules_model.save_model(model)

#visualization
my_modules_model.confusion_matrix( y_test , predictions )
my_modules_model.plot_precision_recall_vs_threshold(x_test , y_test , model)
my_modules_model.plot_roc_curve(x_test , y_test ,predictions, model)

