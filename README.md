# Deep Learning-Based Classification of Tea Leaf Diseases Using EfficientNet-B0

## Abstract

This study presents a deep learning approach for automated classification of tea leaf diseases using a transfer learning-based EfficientNet-B0 architecture. The proposed model achieved an overall validation accuracy of **99.17%** on a dataset comprising six distinct disease classes. The system demonstrates exceptional performance across all disease categories, with the healthy class achieving perfect classification metrics (100% precision, recall, and F1-score).

## 1. Introduction

Tea leaf disease classification is critical for early detection and management of plant health issues. This research employs state-of-the-art deep learning techniques to automate the identification of multiple tea leaf disease conditions, enabling rapid and accurate diagnosis.

## 2. Methodology

### 2.1 Dataset

The dataset consists of 9,000 augmented images distributed across six classes:
- Brown Bright (1,500 images)
- Grey Bright (1,500 images)
- Healthy (1,500 images)
- Helioptis (1,500 images)
- Leaf Scorch (1,500 images)
- Red Rust (1,500 images)

The dataset was partitioned into training (80%) and validation (20%) sets, resulting in 1,800 validation samples for evaluation.

### 2.2 Model Architecture

The classification model is based on EfficientNet-B0, a lightweight yet powerful convolutional neural network architecture. The model utilizes transfer learning from ImageNet pretrained weights, with a custom classification head adapted for six disease classes. The network processes input images of size 224×224 pixels and contains approximately 5.3 million parameters.

### 2.3 Training Configuration

The model was trained for 30 epochs using the following hyperparameters:
- **Optimizer:** AdamW with weight decay of 0.05
- **Initial Learning Rate:** 1×10⁻³
- **Learning Rate Schedule:** Cosine annealing with 10% warmup
- **Batch Size:** 32
- **Loss Function:** CrossEntropyLoss with label smoothing (α = 0.1)

### 2.4 Data Augmentation

To enhance model generalization and prevent overfitting, the following augmentation techniques were applied:
- Random horizontal flipping (probability: 0.5)
- Random rotation (±10 degrees)
- Color jittering (brightness: 0.2, contrast: 0.2)
- MixUp augmentation (α = 0.2, probability: 0.5)
- CutMix augmentation (α = 1.0, probability: 0.5)

### 2.5 Regularization

The training process incorporated several regularization techniques:
- Label smoothing (0.1)
- Early stopping with patience of 20 epochs
- Weight decay (0.05)

## 3. Results

### 3.1 Overall Performance

The proposed model achieved exceptional performance on the validation set:
- **Overall Validation Accuracy: 99.17%**
- **Best Model Performance:** Achieved at epoch 27
- **Macro-Averaged F1-Score: 0.99**
- **Weighted-Averaged F1-Score: 0.99**

### 3.2 Class-wise Performance

The detailed classification performance for each disease class is presented in Table 1.

**Table 1: Class-wise Classification Performance**

| Disease Class | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| Brown Bright  | 0.98      | 0.99   | 0.99     | 309     |
| Grey Bright   | 0.98      | 0.99   | 0.98     | 280     |
| Healthy       | **1.00**  | **1.00** | **1.00** | 293     |
| Helioptis     | 1.00      | 0.99   | 0.99     | 294     |
| Leaf Scorch   | 0.99      | 0.99   | 0.99     | 315     |
| Red Rust      | 0.99      | 0.99   | 0.99     | 309     |
| **Macro Average** | **0.99** | **0.99** | **0.99** | **1,800** |
| **Weighted Average** | **0.99** | **0.99** | **0.99** | **1,800** |

### 3.3 Key Findings

1. **Perfect Classification of Healthy Leaves:** The model achieved 100% precision, recall, and F1-score for the healthy class, demonstrating excellent capability in distinguishing healthy leaves from diseased ones.

2. **Consistent High Performance:** All disease classes achieved precision and recall values ≥98%, indicating robust classification performance across all categories.

3. **Balanced Performance:** The model demonstrates excellent balance between precision and recall across all classes, with F1-scores consistently above 0.98.

4. **Low Misclassification Rate:** The overall accuracy of 99.17% corresponds to a misclassification rate of less than 1%, indicating high reliability for practical applications.

### 3.4 Training Dynamics

The model training progressed smoothly over 30 epochs:
- **Final Training Accuracy:** 96.58%
- **Final Validation Loss:** 0.1252
- **Best Validation Accuracy:** 99.17% (Epoch 27)
- The model demonstrated stable convergence with no signs of overfitting

## 4. Discussion

The results demonstrate that the EfficientNet-B0 architecture, combined with appropriate data augmentation and regularization techniques, is highly effective for tea leaf disease classification. The model's ability to achieve near-perfect classification of healthy leaves (100% across all metrics) is particularly noteworthy, as it suggests the system can reliably distinguish between healthy and diseased conditions.

The consistent high performance across all disease classes (≥98% precision and recall) indicates that the model has learned robust feature representations that generalize well to the validation set. The balanced precision-recall trade-off suggests the model is well-calibrated and suitable for practical deployment.

## 5. Conclusion

This study successfully demonstrates the application of deep learning for automated tea leaf disease classification, achieving an overall accuracy of 99.17%. The proposed EfficientNet-B0-based model shows exceptional performance across all six disease classes, with particular strength in identifying healthy leaves. These results suggest that the model is suitable for real-world applications in agricultural monitoring and disease management systems.

## 6. Visualizations

The following visualizations are available for research paper inclusion:

- **research_results.png:** Publication-quality figure showing overall accuracy and class-wise performance metrics
- **training_curves.png:** Training and validation loss/accuracy curves over 30 epochs
- **confusion_matrix.png:** Confusion matrix visualization (raw counts and normalized percentages)

## References

[Add relevant references here]

---

**Note:** For detailed implementation and code, please refer to the source files in this repository. The trained model weights are available in `best_model.pth`.
# fficientNet-B0
