import random

# Standard libraries
import random
import numpy as np
import pandas as pd

# Machine Learning and Deep Learning libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, confusion_matrix

# Visualization libraries
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.io as pio

# Set plotly renderer if needed
pio.renderers.default = "browser"  # Use "notebook" if you’re in Jupyter or "browser" for VS Code


# Define amino acids and synthetic peptide properties
amino_acids = "ACDEFGHIKLMNPQRSTVWY"
num_sequences = 1000  # Number of synthetic peptides
min_length = 10       # Minimum length of peptide
max_length = 30       # Maximum length of peptide

# Generate random peptide sequences
synthetic_peptides = [''.join(random.choices(amino_acids, k=random.randint(min_length, max_length)))
                      for _ in range(num_sequences)]

print("Sample synthetic peptides:", synthetic_peptides[:5])


sequence_length = 5  # Number of amino acids to look back for prediction

# Create input-output pairs
inputs = []
outputs = []

for peptide in synthetic_peptides:
    for i in range(len(peptide) - sequence_length):
        input_seq = peptide[i:i + sequence_length]
        output_aa = peptide[i + sequence_length]
        inputs.append(input_seq)
        outputs.append(output_aa)

print("Sample input sequences:", inputs[:5])
print("Sample outputs:", outputs[:5])


# Encode amino acids as integers
encoder = LabelEncoder()
encoder.fit(list(amino_acids))
encoded_inputs = [encoder.transform(list(seq)) for seq in inputs]
encoded_outputs = encoder.transform(outputs)

# One-hot encode the input and output sequences
X = np.array([to_categorical(seq, num_classes=len(amino_acids)) for seq in encoded_inputs])
y = to_categorical(encoded_outputs, num_classes=len(amino_acids))

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Define the LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(sequence_length, len(amino_acids)), return_sequences=False))
model.add(Dense(len(amino_acids), activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2)

# Function to predict the next amino acid in a sequence
def predict_next_amino_acid(input_seq):
    encoded_input = to_categorical(encoder.transform(list(input_seq)), num_classes=len(amino_acids))
    encoded_input = np.expand_dims(encoded_input, axis=0)
    prediction = model.predict(encoded_input)
    predicted_aa = encoder.inverse_transform([np.argmax(prediction)])
    return predicted_aa[0]

# Let's create a sample of test sequences and their true next amino acids
test_inputs = X[:100]  # Adjust the number as needed to fit available data
test_labels = y[:100]  # True next amino acid labels for test sequences

# Generate predictions
predictions = model.predict(test_inputs)

# Convert predictions and labels to one-hot encoded format for metrics calculation
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(test_labels, axis=1)

# AUC for each class (one-vs-rest)
auc_scores = {}
for i, amino_acid in enumerate(encoder.classes_):
    # Extract true labels and predicted probabilities for this amino acid
    true_binary = (true_classes == i).astype(int)
    pred_binary = predictions[:, i]
    
    # Calculate AUC score
    auc = roc_auc_score(true_binary, pred_binary)
    auc_scores[amino_acid] = auc

print("AUC Scores for each amino acid:", auc_scores)

# Initialize dictionaries to store sensitivity and specificity for each class
sensitivity = {}
specificity = {}

# Compute confusion matrix
cm = confusion_matrix(true_classes, predicted_classes, labels=range(len(amino_acids)))

# Calculate sensitivity and specificity for each class
for i, amino_acid in enumerate(encoder.classes_):
    # True Positives, False Positives, False Negatives, True Negatives
    TP = cm[i, i]
    FN = cm[i, :].sum() - TP
    FP = cm[:, i].sum() - TP
    TN = cm.sum() - (TP + FN + FP)

    # Sensitivity (Recall)
    sensitivity[amino_acid] = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    # Specificity
    specificity[amino_acid] = TN / (TN + FP) if (TN + FP) > 0 else 0

# Create a DataFrame from the metrics dictionaries
metrics_df = pd.DataFrame({
    "Amino Acid": list(auc_scores.keys()),
    "AUC": list(auc_scores.values()),
    "Sensitivity": list(sensitivity.values()),
    "Specificity": list(specificity.values())
})

# Set Amino Acid as the index for a cleaner view
metrics_df.set_index("Amino Acid", inplace=True)
print(metrics_df)

# Save the metrics DataFrame to a CSV file
metrics_df.to_csv("amino_acid_metrics.csv")

# Display download link in Colab (if you're using Google Colab)
try:
    from google.colab import files
    files.download("amino_acid_metrics.csv")
except ImportError:
    print("CSV file saved as 'amino_acid_metrics.csv'. Please download manually if not in Colab.")


import plotly.express as px
import plotly.graph_objects as go

# Reshape the DataFrame for easy plotting with Plotly Express
metrics_long_df = metrics_df.reset_index().melt(id_vars="Amino Acid", 
                                                var_name="Metric", 
                                                value_name="Score")

# Create an interactive bar plot
fig = px.bar(metrics_long_df, 
             x="Amino Acid", 
             y="Score", 
             color="Metric", 
             barmode="group",
             title="Amino Acid Prediction Metrics (AUC, Sensitivity, Specificity)")

# Customize layout
fig.update_layout(xaxis_title="Amino Acid", 
                  yaxis_title="Metric Score",
                  legend_title_text="Metric",
                  width=900,
                  height=500)

fig.show()

import plotly.figure_factory as ff

# Convert DataFrame to a format suitable for Plotly's heatmap
heatmap_fig = go.Figure(data=go.Heatmap(
    z=metrics_df.values,
    x=metrics_df.columns,
    y=metrics_df.index,
    colorscale="YlGnBu",
    colorbar=dict(title="Score"),
))

# Update layout for readability
heatmap_fig.update_layout(
    title="Heatmap of Amino Acid Prediction Metrics",
    xaxis_title="Metric",
    yaxis_title="Amino Acid",
    width=800,
    height=500
)

heatmap_fig.show()
