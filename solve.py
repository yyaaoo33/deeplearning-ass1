import torch
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

seed = 57
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def preprocess(df):

    zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for column in zero_columns:
        df[column] = df[column].replace(0, np.nan)
    df = df.fillna(df.mean())

    df['Glucose_to_Insulin_Ratio'] = df['Glucose'] / (df['Insulin'] + 1)  
    df['BMI_Category'] = pd.cut(df['BMI'], bins=[0, 18.5, 25, 30, 100], labels=[0, 1, 2, 3])

    df = pd.get_dummies(df, columns=['BMI_Category'])

    scaler = StandardScaler()
    numerical_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Glucose_to_Insulin_Ratio']
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    return df

class DiabetesModel(torch.nn.Module):
    def __init__(self, num_features):
        super(DiabetesModel, self).__init__()
        self.layer1 = torch.nn.Linear(num_features, 72)
        self.layer2 = torch.nn.Linear(72, 64)
        self.layer3 = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return torch.sigmoid(self.layer3(x)).squeeze()

def train(model, X, y, learning_rate=0.001, epochs=200):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCELoss()

    losses = []  
    for _ in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())  

    return losses  

df = pd.read_csv('diabetes.csv')
df = preprocess(df)

X = df.drop('Outcome', axis=1)
y = df['Outcome']

print(f"Data shape: {X.shape}")
print(f"Feature means:\n{X.mean()}")
print(f"Feature standard deviations:\n{X.std()}")
print(f"Class distribution:\n{y.value_counts()}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

X_train = torch.FloatTensor(X_train.values)
y_train = torch.FloatTensor(y_train.values)
X_test = torch.FloatTensor(X_test.values)
y_test = torch.FloatTensor(y_test.values)

num_features = X_train.shape[1]
model = DiabetesModel(num_features=num_features)

losses = train(model, X_train, y_train, learning_rate=0.001, epochs=40)

plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('loss_curve.png')
plt.show()

with torch.no_grad():
    y_pred = (model(X_test) > 0.5).float().numpy()

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png')
plt.show()

from sklearn.metrics import classification_report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

y_pred_proba = model(X_test).detach().numpy()
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')
plt.show()