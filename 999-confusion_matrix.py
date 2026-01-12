import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

path = 'metrics_out\\confusion_matrix.csv'
# print(os.path.dirname(path))
# Read CSV file
data = pd.read_csv(path, index_col=0)

# Extract class names
classes = data.columns.tolist()

# Convert to NumPy array
confusion_matrix = data.values

# Create heatmap
plt.figure(figsize=(24, 20))
sns.heatmap(confusion_matrix, annot=True, fmt='g', cmap='Blues', 
            xticklabels=classes, yticklabels=classes)

# Set title and labels
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Rotate labels
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels by 45 degrees
plt.yticks(rotation=0)  # Do not rotate y-axis labels

# Save image with 300 dpi resolution
cm_img_path = os.path.join(os.path.dirname(path),'confusion_matrix.png')
plt.savefig(cm_img_path, dpi=300)




# Calculate overall accuracy
correct_predictions = np.sum(np.diag(confusion_matrix))  # Sum of diagonal elements
total_samples = np.sum(confusion_matrix)  # Number of all samples
overall_accuracy = correct_predictions / total_samples  # Calculate overall accuracy
print(correct_predictions ,total_samples)
print(f"Overall accuracy is {overall_accuracy:.4f}")  # Output result, keeping two decimal places

# Store precision and recall for each class
precision_list = []
recall_list = []

for i in range(len(classes)):
    tp = confusion_matrix[i, i]  # True Positive
    fp = np.sum(confusion_matrix[:, i]) - tp  # False Positive
    fn = np.sum(confusion_matrix[i, :]) - tp  # False Negative
    
    # Calculate precision and recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    precision_list.append(precision)
    recall_list.append(recall)

# Calculate macro average, micro average and weighted average
macro_precision = np.mean(precision_list)  # Macro Average
macro_recall = np.mean(recall_list)        # Macro Average

micro_precision = np.sum(precision_list) / len(classes)  # Micro Average
micro_recall = np.sum(recall_list) / len(classes)        # Micro Average

weighted_precision = np.average(precision_list, weights=np.sum(confusion_matrix, axis=1))  # Weighted Average
weighted_recall = np.average(recall_list, weights=np.sum(confusion_matrix, axis=1))        # Weighted Average

# Output averages
print(f"Macro Average Precision: {macro_precision:.4f}, Macro Average Recall: {macro_recall:.4f}")
print(f"Micro Average Precision: {micro_precision:.4f}, Micro Average Recall: {micro_recall:.4f}")
print(f"Weighted Average Precision: {weighted_precision:.4f}, Weighted Average Recall: {weighted_recall:.4f}")
