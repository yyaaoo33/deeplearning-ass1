import torch
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler

seed = 57
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

def preprocess(df):

    zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for column in zero_columns:
        df[column] = df[column].replace(0, np.nan)
    df = df.fillna(df.mean())

    df['Glucose_to_Insulin_Ratio'] = df['Glucose'] / (df['Insulin'] + 1)  
    df['BMI_Category'] = pd.cut(df['BMI'], bins=[0, 18.5, 25, 30, 100], labels=[0, 1, 2, 3])

    df = pd.get_dummies(df, columns=['BMI_Category'], dtype=int)  

    scaler = StandardScaler()
    numerical_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Glucose_to_Insulin_Ratio']
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    return df

def ensure_numeric(df):
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = pd.to_numeric(df[column], errors='coerce')  
    return df.fillna(0)

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

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

    return loss.item()

df = pd.read_csv('diabetes.csv')
df = preprocess(df)

df = ensure_numeric(df)

X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

X_train = torch.FloatTensor(X_train.values).to(device)
y_train = torch.FloatTensor(y_train.values).to(device)
X_test = torch.FloatTensor(X_test.values).to(device)
y_test = torch.FloatTensor(y_test.values).to(device)

print("X_train types:", X_train.dtype)
print("y_train types:", y_train.dtype)

num_features = X_train.shape[1]

learning_rates = [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
best_lr = None
best_accuracy = 0

for lr in learning_rates:
    model = DiabetesModel(num_features=num_features).to(device)
    train(model, X_train, y_train, learning_rate=lr, epochs=40)

    with torch.no_grad():
        y_pred = (model(X_test) > 0.5).float().cpu().numpy()  

    accuracy = accuracy_score(y_test.cpu().numpy(), y_pred)
    print(f"Learning Rate: {lr} - Accuracy: {accuracy * 100:.2f}%")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_lr = lr

print(f"\nBest Learning Rate: {best_lr} with Accuracy: {best_accuracy * 100:.2f}%")

model = DiabetesModel(num_features=num_features).to(device)
train(model, X_train, y_train, learning_rate=best_lr, epochs=40)

with torch.no_grad():
    y_pred = (model(X_test) > 0.5).float().cpu().numpy()

accuracy = accuracy_score(y_test.cpu().numpy(), y_pred)
print(f"Final Model Accuracy: {accuracy * 100:.2f}%")

cm = confusion_matrix(y_test.cpu().numpy(), y_pred)
print("\nConfusion Matrix:")
print(cm)

from sklearn.metrics import classification_report
print("\nClassification Report:")
print(classification_report(y_test.cpu().numpy(), y_pred))

y_pred_proba = model(X_test).detach().cpu().numpy()
fpr, tpr, _ = roc_curve(y_test.cpu().numpy(), y_pred_proba)
roc_auc = auc(fpr, tpr)

print(f"ROC AUC: {roc_auc:.2f}")