# Let's assume that X is your feature matrix and y are your labels
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import numpy as np
from sklearn.metrics import classification_report

observedEmotions = ['sad', 'fearful', 'disgust', 'happy', 'angry']
# Load x and y from the saved files
with open('x_radvess (540).pkl', 'rb') as f:
    x = pickle.load(f)

with open('y_radvess (540).pkl', 'rb') as f:
    y = pickle.load(f)

# Define the model
model = MLPClassifier(alpha=1, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,),learning_rate='adaptive', max_iter=1000, random_state=42)

# Define the RFE model using ExtraTreesClassifier as the estimator
rfe_model = ExtraTreesClassifier(n_estimators=100, random_state=42)

# Create the RFE object and rank each feature
rfe = RFE(estimator=rfe_model, n_features_to_select=147) # Adjust number of features to select

# Create a pipeline to combine the steps
pipeline = Pipeline(steps=[('s',rfe),('m',model)])

# Apply standard scaling to the features
scaler = StandardScaler()
x = scaler.fit_transform(x)

# Fit the pipeline
pipeline.fit(x, y)

# You can check which features have been selected using
print(rfe.support_)

# And the ranking of the features
print(rfe.ranking_)

# Transform the data
X_transformed = rfe.transform(x)
x_train, x_test, y_train, y_test=train_test_split(np.array(X_transformed), y, test_size=0.25, random_state=42)

model.fit(x_train, y_train)
print((x_train.shape[0], x_test.shape[0]))
print(f'Features extracted:{x_train.shape[1]}')

y_pred=model.predict(x_test)
accuracy= accuracy_score(y_true=y_test, y_pred=y_pred)

print("Accuracy: {:.2f}%".format(accuracy*100))

report = classification_report(y_test, y_pred, target_names=observedEmotions, output_dict=True)
weighted_precision = report['weighted avg']['precision']
weighted_recall = report['weighted avg']['recall']
weighted_f1_score = report['weighted avg']['f1-score']

print("Weighted Precision: {:.2f}%".format(weighted_precision*100))
print("Weighted Recall: {:.2f}%".format(weighted_recall*100))
print("Weighted F1-score: {:.2f}%".format(weighted_f1_score*100))
