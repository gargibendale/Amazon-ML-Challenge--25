## Amazon ML Challenge 2025 – Product Price Prediction (Multimodal Regression)

### Overview
This project was developed for the Amazon ML Challenge 2025. The task was to **predict product prices** using product catalog **text content and the corresponding product image**. The evaluation metric was **SMAPE (Symmetric Mean Absolute Percentage Error)**.
It is a regression problem involving both textual and visual data, requiring **effective multimodal feature extraction and modeling.**

### Problem statement
Given a product’s catalog information (item name, description, brand, bullet points, and quantity details) and its image, predict the product’s price.

## Approach  

### I. Exploratory Data Analysis (EDA)  

#### 1. Image Analysis  
- Checked the **lowest**, **highest**, and **average** image dimensions.  
- Found that many text samples had **duplicate image files**, explaining the mismatch in counts between text and image datasets.  
- Identified **one missing** and **one corrupted** image file in the training set.  
- Used **dummy (zero) embeddings** for missing or corrupted images while generating embeddings.  

#### 2. Text Analysis  
- Extracted useful columns from the product catalog:
  - **description**
  - **item name**
  - **brand name**
  - **bullet points**
  - **value** (numeric amount or quantity)
  - **unit** (e.g., ounce, fl. oz., etc.)
- Applied **rigorous preprocessing**:
  - Removed punctuations and unnecessary symbols.  
  - Dropped missing values for important columns like bullet points.  
  - Removed the *description* column due to over two-thirds missing data.  
  - Detected and removed **outliers** using the IQR method on numeric fields (*price* and *value*).  
- The final cleaned training dataset contained **56,102 samples**.  

---

### II. Feature Representation (Embeddings)  

#### 1. Image Embeddings  
- Used **pretrained models** from Keras for visual feature extraction:  
  - **EfficientNetV2-S**  
  - **EfficientNetB7**  
  - **ResNet50**  
  - **EfficientNetV2B0** (performed best overall)  
  - **CLIP BLIP**  
- Extracted **feature vectors** from the penultimate layer of each model.  
- Stored embeddings to disk for later use during multimodal training.  

#### 2. Text Embeddings  
- Started with **TF-IDF** vectorization for baseline text representation.  
- Applied **SVD truncation** for dimensionality reduction and noise removal.  
- Later experimented with **BERT embeddings** for contextual understanding.  
- Fine-tuned **BERT** on the product catalog text using **PEFT (LoRA)** for parameter-efficient adaptation.  

---

#### Error Metric - SMAPE (Symmetric Mean Absolute Percentage Error)
SMAPE is used in regression problems because it provides a symmetric and relative error measure, making it useful for comparing models across different scales, and it is less sensitive to the direction of error than MAPE. It calculates the percentage error relative to the average of the actual and predicted values, which helps standardize results and allows for easier interpretation and comparison.  

![SMAPE](https://github.com/user-attachments/assets/4654b07e-684d-420b-a237-cec27941aea4)

### III. Model Training  

We experimented with several model and feature combinations:

| Combination | Features Used | Model(s) | Notes |
|--------------|----------------|-----------|--------|
| 1 | TF-IDF (text only) | Linear Regression, XGBoost | Baseline performance |
| 2 | TF-IDF (text) + EfficientNetV2S (image) | XGBoost | Moderate improvement |
| 3 | BERT (text only) | Regression Head | Transformer-based features |
| 4 | TF-IDF (text) + EfficientNetV2B0 (image) | Ensemble (XGBoost + LightGBM + CatBoost) | **Best performing combination** |

and many more..

- The **best validation SMAPE (~45%)** was achieved using **TF-IDF (text)** + **EfficientNetV2B0 (image)** embeddings trained with an **ensemble** of gradient boosting models.  
- GPU limitations (Kaggle’s free GPU resources) restricted further fine-tuning and rapid experimentation on additional model combinations and advanced architectures.
- Despite that, the project provided strong insights into **multimodal regression** and was a great learning experience overall.

_(Note: The final outputs from the notebook were lost during cleanup, so only summary-level results are available.)_

## Technologies and Libraries
- Python, NumPy, Pandas, Scikit-learn
- TensorFlow / Keras, PyTorch, Transformers (Hugging Face), Kaggle GPUs
- XGBoost, LightGBM, CatBoost
- PEFT (LoRA) for parameter-efficient fine-tuning + Weights&Biases(W&B)

## Future improvements
1 - Train on higher-quality GPUs for better multimodal model convergence.  
2 - Try contrastive multimodal pretraining (CLIP-style fusion).  
3 - Use model interpretability techniques to analyze which features affect price the most.  
4 - Apply better text preprocessing techniques and include more structured metadata (category, brand, etc.) for boosting accuracy.

## Team members
- [@gargibendale](https://github.com/gargibendale)
- [@AakashHubGit](https://github.com/aakashhubgit)
- [@sidheshsahu](https://github.com/sidheshsahu)
- Harshada Sutar

