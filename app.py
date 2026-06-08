import streamlit as st
import torch
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score


class SimpleCNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)

        # Dummy tensor to calculate the output size after conv and pool layers
        dummy_input = torch.zeros(1, 1, input_size)
        self.flattened_size = self._get_flattened_size(dummy_input)

        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def _get_flattened_size(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        return x.numel()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

@st.cache_resource
def load_model(model_path):
    model = SimpleCNN(input_size=100, num_classes=6)  
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

model = load_model('/Users/jade/Downloads/best_model_CNN.pth')

st.title('Human Activity Recognition')

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    test_data = pd.read_csv(uploaded_file)

    # Preprocess the test data
    scaler = StandardScaler()
    pca = PCA(n_components=100)

    features = test_data.drop(columns=['Activity','subject'])
    scaled_features = scaler.fit_transform(features)
    pca_features = pca.fit_transform(scaled_features)

    # Encode the 'Activity' labels
    le = LabelEncoder()
    test_data['Activity'] = le.fit_transform(test_data['Activity'])

    X_test = torch.tensor(pca_features, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(test_data['Activity'].values, dtype=torch.long)

    # Perform classification
    with torch.no_grad():
        outputs = model(X_test)
        predictions = torch.argmax(outputs, dim=1)

    # Print classification report and confusion matrix
    st.write('Classification Report:')
    report = classification_report(y_test, predictions, target_names=le.classes_, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

    st.write('Confusion Matrix:')
    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)

    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)
