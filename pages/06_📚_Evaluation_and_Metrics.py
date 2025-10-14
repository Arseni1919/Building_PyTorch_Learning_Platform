"""
Topic 6: Evaluation & Metrics - Basic Level
"""

import streamlit as st

# Page config
st.set_page_config(
    page_title="06 - Evaluation & Metrics",
    page_icon="✅",
    layout="wide"
)

# Main content
st.markdown("""
# Evaluation & Metrics ✅

## Why Accuracy Isn't Enough

Imagine a fraud detection system where 99% of transactions are legitimate:

```python
# Terrible model that predicts "legitimate" for everything
def bad_model(transaction):
    return "legitimate"  # Always!

# Accuracy: 99%! 🎉 ...but completely useless for catching fraud!
```

**The problem**: Accuracy is misleading for imbalanced datasets. We need better metrics!

---

## Classification Metrics Overview

For classification tasks, we have several important metrics:

### 1. Accuracy

**Formula**: `(Correct Predictions) / (Total Predictions)`

```python
correct = (predictions == targets).sum().item()
accuracy = correct / len(targets)
print(f"Accuracy: {accuracy * 100:.2f}%")
```

**When to use**: Balanced datasets (each class has similar number of samples)
**When NOT to use**: Imbalanced datasets (rare classes)

---

### 2. Precision, Recall, and F1-Score

These metrics are crucial for understanding performance per class!

#### Precision: "How many of my positive predictions were correct?"

**Formula**: `True Positives / (True Positives + False Positives)`

**Example**: Medical diagnosis
- Model predicts 100 patients have disease
- 90 actually have it, 10 don't
- **Precision = 90/100 = 90%**

```python
from sklearn.metrics import precision_score

precision = precision_score(y_true, y_pred, average='weighted')
print(f"Precision: {precision:.4f}")
```

#### Recall: "How many actual positives did I find?"

**Formula**: `True Positives / (True Positives + False Negatives)`

**Example**: Medical diagnosis
- 100 patients actually have disease
- Model correctly identifies 90
- **Recall = 90/100 = 90%**

```python
from sklearn.metrics import recall_score

recall = recall_score(y_true, y_pred, average='weighted')
print(f"Recall: {recall:.4f}")
```

#### F1-Score: Harmonic mean of precision and recall

**Formula**: `2 × (Precision × Recall) / (Precision + Recall)`

**Why harmonic mean?** It penalizes extreme values (low precision OR low recall).

```python
from sklearn.metrics import f1_score

f1 = f1_score(y_true, y_pred, average='weighted')
print(f"F1-Score: {f1:.4f}")
```

---

### 3. Confusion Matrix

A table showing all predictions vs actual labels:

```python
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Get predictions
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Create confusion matrix
cm = confusion_matrix(all_labels, all_preds)

# Plot
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
```

**How to read it**:
- Diagonal = correct predictions
- Off-diagonal = errors
- Each row = actual class
- Each column = predicted class

Example output:
```
            Predicted
           0    1    2
Actual 0  95   3    2     (Class 0: 95 correct, 5 wrong)
       1   1  88   11    (Class 1: 88 correct, 12 wrong)
       2   4   9   87    (Class 2: 87 correct, 13 wrong)
```

---

## Complete Evaluation Function

Here's a comprehensive evaluation function with all metrics:

```python
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

def evaluate_model(model, test_loader, device, class_names=None):
    \"\"\"Comprehensive model evaluation with all metrics\"\"\"
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Get predictions
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            # Store results
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    # Print results
    print(f"{'='*50}")
    print(f"MODEL EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"{'='*50}")

    # Detailed classification report
    if class_names:
        print("\\nPer-Class Metrics:")
        print(classification_report(all_labels, all_preds,
                                   target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\\nConfusion Matrix:")
    print(cm)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }

# Usage
class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

results = evaluate_model(model, test_loader, device, class_names)
```

Example output:
```
==================================================
MODEL EVALUATION RESULTS
==================================================
Accuracy:  0.8742 (87.42%)
Precision: 0.8756
Recall:    0.8742
F1-Score:  0.8745
==================================================

Per-Class Metrics:
              precision    recall  f1-score   support

     T-shirt       0.82      0.88      0.85      1000
     Trouser       0.98      0.96      0.97      1000
    Pullover       0.82      0.85      0.83      1000
       Dress       0.89      0.89      0.89      1000
        Coat       0.82      0.85      0.84      1000
      Sandal       0.96      0.93      0.95      1000
       Shirt       0.71      0.64      0.67      1000
     Sneaker       0.93      0.95      0.94      1000
         Bag       0.96      0.96      0.96      1000
  Ankle boot       0.95      0.94      0.94      1000
```

**Key insights from this report**:
- Shirt has lowest performance (precision 0.71) - often confused with T-shirt/Coat
- Trouser and Bag have highest performance (>96%)
- Overall balanced performance across classes

---

## Per-Class Analysis

```python
def analyze_worst_classes(results, class_names):
    \"\"\"Find classes with lowest F1 scores\"\"\"
    from sklearn.metrics import f1_score

    per_class_f1 = []
    for i in range(len(class_names)):
        # Binary mask for this class
        true_binary = (results['labels'] == i)
        pred_binary = (results['predictions'] == i)

        # Calculate F1 for this class
        f1 = f1_score(true_binary, pred_binary)
        per_class_f1.append((class_names[i], f1))

    # Sort by F1 (lowest first)
    per_class_f1.sort(key=lambda x: x[1])

    print("Classes needing improvement:")
    for name, f1 in per_class_f1[:3]:
        print(f"  {name}: F1={f1:.4f}")

analyze_worst_classes(results, class_names)
```

Output:
```
Classes needing improvement:
  Shirt: F1=0.6724
  Pullover: F1=0.8341
  Coat: F1=0.8392
```

**Why this matters**: You can focus training/data collection on weak classes!

---

## Regression Metrics

For predicting continuous values (house prices, temperatures, etc.):

### 1. Mean Squared Error (MSE)

```python
mse = torch.nn.functional.mse_loss(predictions, targets)
print(f"MSE: {mse.item():.4f}")
```

**Interpretation**: Average squared difference. Lower is better. Penalizes large errors heavily.

### 2. Root Mean Squared Error (RMSE)

```python
rmse = torch.sqrt(mse)
print(f"RMSE: {rmse.item():.4f}")
```

**Interpretation**: Same units as target variable. Easier to interpret than MSE.

### 3. Mean Absolute Error (MAE)

```python
mae = torch.nn.functional.l1_loss(predictions, targets)
print(f"MAE: {mae.item():.4f}")
```

**Interpretation**: Average absolute difference. Less sensitive to outliers than MSE.

---

## Model Confidence Analysis

```python
def analyze_confidence(model, test_loader, device):
    \"\"\"Analyze prediction confidence\"\"\"
    model.eval()
    confidences = []

    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)

            # Get max probability (confidence)
            max_probs, _ = torch.max(probs, dim=1)
            confidences.extend(max_probs.cpu().numpy())

    confidences = np.array(confidences)

    print(f"Average confidence: {confidences.mean():.4f}")
    print(f"Min confidence: {confidences.min():.4f}")
    print(f"Max confidence: {confidences.max():.4f}")

    # Find low-confidence predictions (uncertain)
    low_conf = confidences < 0.5
    print(f"Low confidence predictions: {low_conf.sum()} "
          f"({low_conf.sum()/len(confidences)*100:.2f}%)")

analyze_confidence(model, test_loader, device)
```

**Why this matters**: Low confidence predictions might need human review in production!

---

## Connection to Transformers

Modern LLMs use these same evaluation metrics:

- **Perplexity**: Main metric for language models (measures prediction confidence)
- **BLEU/ROUGE**: For text generation quality
- **Accuracy**: For classification tasks (sentiment, NER, etc.)

Example LLM evaluation:
```python
# Evaluate GPT-style model
def evaluate_lm(model, val_loader):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            outputs = model(batch['input_ids'])
            loss = criterion(outputs.view(-1, vocab_size),
                           batch['labels'].view(-1))
            total_loss += loss.item()

    # Perplexity = exp(average loss)
    perplexity = torch.exp(torch.tensor(total_loss / len(val_loader)))
    return perplexity
```

---

## Key Takeaways 💡

✅ **Accuracy alone is misleading** for imbalanced datasets
✅ Use **precision, recall, F1** for per-class performance
✅ **Confusion matrix** shows exactly where your model fails
✅ Always evaluate on a **separate test set** (never training data!)
✅ Analyze **per-class performance** to find weak spots
✅ Check **prediction confidence** to identify uncertain predictions

**Congratulations!** You've completed the Basic Level! 🎉 You now understand:
- PyTorch tensors and autograd
- Building neural networks with nn.Module
- Loss functions and optimizers
- Training and evaluation
- Comprehensive model metrics

---

## Next Steps 🚀

**You're ready for Intermediate Level!** Move on to learn about:
- Custom datasets and data loaders
- Convolutional Neural Networks (CNNs)
- Batch normalization and dropout
- Transfer learning
- And more!
""")

# Quiz section
st.markdown("---")
st.markdown("## 📝 Knowledge Check")

questions = [
    {
        "question": "Why is accuracy not enough for imbalanced datasets?",
        "options": [
            "It takes too long to calculate",
            "It can be misleading - a model predicting only the majority class can have high accuracy",
            "It doesn't work with neural networks",
            "It requires too much memory"
        ],
        "correct": "It can be misleading - a model predicting only the majority class can have high accuracy",
        "explanation": "For imbalanced datasets (e.g., 99% class A, 1% class B), a model that always predicts class A gets 99% accuracy but is useless for detecting class B. That's why we need metrics like precision, recall, and F1-score."
    },
    {
        "question": "What does a confusion matrix show?",
        "options": [
            "Only the incorrect predictions",
            "The training loss over time",
            "All predictions vs actual labels in a table format",
            "The model's confidence scores"
        ],
        "correct": "All predictions vs actual labels in a table format",
        "explanation": "A confusion matrix shows how many samples of each actual class were predicted as each possible class. The diagonal shows correct predictions, and off-diagonal cells show specific types of errors."
    },
    {
        "question": "When analyzing a confusion matrix, where do correct predictions appear?",
        "options": [
            "In the top row",
            "In the right column",
            "On the diagonal (top-left to bottom-right)",
            "In the bottom-left corner"
        ],
        "correct": "On the diagonal (top-left to bottom-right)",
        "explanation": "Correct predictions appear on the diagonal because that's where the actual class equals the predicted class (e.g., actual=0 and predicted=0, actual=1 and predicted=1, etc.)."
    },
    {
        "question": "What does precision measure?",
        "options": [
            "How many actual positives were found",
            "What proportion of positive predictions were correct",
            "The overall accuracy of the model",
            "The training time"
        ],
        "correct": "What proportion of positive predictions were correct",
        "explanation": "Precision measures what proportion of positive predictions were actually correct. Formula: True Positives / (True Positives + False Positives). High precision means few false positives."
    }
]

for idx, q in enumerate(questions):
    st.markdown(f"### Question {idx + 1}")
    st.markdown(f"**{q['question']}**")

    user_answer = st.radio(
        "Select your answer:",
        options=q["options"],
        key=f"q{idx}",
        index=None
    )

    if st.button(f"Check Answer {idx + 1}", key=f"btn{idx}"):
        if user_answer:
            if user_answer == q["correct"]:
                st.success(f"✅ Correct! {q['explanation']}")
            else:
                st.error(f"❌ Incorrect. {q['explanation']}")
        else:
            st.warning("Please select an answer first!")

    st.markdown("---")

# Navigation
st.success("🎉 **Congratulations!** You've completed all Basic Level topics!")
st.info("👈 Continue to **Intermediate Level** (Topics 07-13) in the sidebar!")
