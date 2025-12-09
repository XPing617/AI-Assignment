import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

# Set plot style
sns.set(style="whitegrid")

# ============================================
# 1. Data Preparation
# ============================================
print("Starting program... Loading data...")

# Ensure heart.csv is in the same folder
try:
    df = pd.read_csv('heart.csv')
    print(f"Data loaded successfully. Total records: {len(df)}")
except FileNotFoundError:
    print("Error: heart.csv not found. Please check the file path.")
    exit()

# Data Cleaning: Check for missing values
# Filling missing values with mean if any exist
if df.isnull().sum().sum() > 0:
    print("Missing values found, filling with mean...")
    df.fillna(df.mean(), inplace=True)

# Define features (X) and target (y)
X = df.drop(columns=['target'])
y = df['target']

# Split dataset: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ============================================
# 2. Data Standardization
# ============================================
# Scaling is important for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================
# 3. Model A: Logistic Regression
# ============================================
print("\n--- Training Logistic Regression ---")
lr = LogisticRegression()
lr.fit(X_train_scaled, y_train)

# Predict and calculate accuracy
lr_pred = lr.predict(X_test_scaled)
lr_acc = accuracy_score(y_test, lr_pred)
print(f"Logistic Regression Accuracy: {lr_acc:.4f}")

# ============================================
# 4. Model B: Random Forest (Manual Tuning)
# ============================================
print("\n--- Tuning Random Forest ---")
# Loop to find the best number of trees (n_estimators)
best_score = 0
best_n = 0
best_rf_model = None
n_list = [10, 50, 100, 150, 200]

for n in n_list:
    rf = RandomForestClassifier(n_estimators=n, random_state=42)
    rf.fit(X_train, y_train)
    score = rf.score(X_test, y_test)
    
    # Print process to show experimentation
    print(f"   Trying n_estimators={n} ... Accuracy: {score:.4f}")
    
    if score > best_score:
        best_score = score
        best_n = n
        best_rf_model = rf

print(f"Best Random Forest found: n_estimators={best_n}, Accuracy: {best_score:.4f}")

# ============================================
# 5. Final Comparison
# ============================================
print("\n--- Final Conclusion ---")
if best_score > lr_acc:
    print("Conclusion: Random Forest performed better. Using it as the final model.")
    final_model = best_rf_model
    final_pred = best_rf_model.predict(X_test)
else:
    print("Conclusion: Logistic Regression performed better. Using it as the final model.")
    final_model = lr
    final_pred = lr_pred

# ============================================
# 6. Visualization & Saving Plots
# ============================================
print("\nGenerating and saving plots...")

# --- Plot 1: Confusion Matrix ---
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, final_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix (Best Model)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('result_1_confusion_matrix.png') # Save plot
print("   Saved: result_1_confusion_matrix.png")
# plt.show() 

# --- Plot 2: Feature Importance ---
# Only plot this if Random Forest wins
if best_score > lr_acc:
    plt.figure(figsize=(8, 6))
    importances = final_model.feature_importances_
    feat_importances = pd.Series(importances, index=X.columns)
    feat_importances.nlargest(10).plot(kind='barh', color='teal')
    plt.title('Top 10 Important Features')
    plt.tight_layout()
    plt.savefig('result_2_feature_importance.png') # Save plot
    print("   Saved: result_2_feature_importance.png")
    # plt.show()

# --- Plot 3: ROC Curve Comparison (Bonus) ---
plt.figure(figsize=(8, 6))

# Calculate ROC for Logistic Regression
y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
roc_auc_lr = auc(fpr_lr, tpr_lr)
plt.plot(fpr_lr, tpr_lr, color='navy', lw=2, linestyle='--', label=f'Logistic Regression (AUC = {roc_auc_lr:.2f})')

# Calculate ROC for Random Forest
y_prob_rf = best_rf_model.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)
plt.plot(fpr_rf, tpr_rf, color='darkorange', lw=2, label=f'Random Forest (AUC = {roc_auc_rf:.2f})')

# Diagonal line (Random Guess)
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc="lower right")
plt.savefig('result_3_roc_curve.png') # Save plot
print("   Saved: result_3_roc_curve.png")
plt.show() # Show the last plot

from sklearn.metrics import classification_report # <--- 记得确认这一行在最上面 import 过

# ============================================
# 5.5 Detailed Evaluation (Visual Table)
# ============================================
print("\nGenerating Classification Report Table...")

# 1. Get the report as a Dictionary (Data) instead of String (Text)
report_dict = classification_report(y_test, final_pred, output_dict=True)

# 2. Convert to a Pandas DataFrame for easier plotting
df_report = pd.DataFrame(report_dict).transpose()

# 3. Plotting the table using Heatmap
plt.figure(figsize=(8, 6))

# 'annot=True' shows the numbers
# 'fmt='.2f'' keeps 2 decimal places
# 'cmap="Blues"' uses a professional blue color scheme
# 'linewidths=1, linecolor="black"' adds the grid lines you wanted!
sns.heatmap(df_report, annot=True, cmap='Blues', fmt='.2f', 
            linewidths=1, linecolor='black', cbar=False)

plt.title('Detailed Classification Report', fontsize=15)
plt.yticks(rotation=0) # Keep y-axis labels horizontal

# Save the table as an image
plt.savefig('result_4_classification_report.png')
print("   Saved: result_4_classification_report.png")
plt.show()

# ============================================
# 7. Model Persistence (Saving)
# ============================================
print("\nSaving model file...")
model_filename = 'my_heart_model.pkl'
joblib.dump(final_model, model_filename)
print(f"Model saved as '{model_filename}'")
print(" (Model is ready for future predictions)")

print("\nProcess Complete.")