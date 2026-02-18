# Comparative Sentiment Analysis on Persian Food Reviews

This repository presents a comprehensive comparative study of various NLP architectures for Sentiment Analysis on the **SnappFood dataset**. The project evaluates five different modeling approaches, ranging from classical frequency-based methods to modern Transformer-based architectures, to identify the most effective strategy for Persian text classification.

## 📊 Project Overview
The goal of this project is to classify customer reviews as either "Happy" (Positive) or "Sad" (Negative). This involves navigating the unique challenges of the Persian language, including informal "Pinglish" variations, right-to-left (RTL) formatting, and complex morphology.

## 🚀 Models Implemented
1.  **ParsBERT (Transformer):** Fine-tuned a pre-trained Persian BERT model (from HooshvareLab) using the Hugging Face `transformers` library.
2.  **LSTM (Deep Learning):** A Recurrent Neural Network architecture designed to capture sequential dependencies in text.
3.  **Logistic Regression:** A high-performing baseline using TF-IDF vectorization.
4.  **Random Forest:** An ensemble method to test non-linear decision boundaries on word frequencies.
5.  **Multinomial Naive Bayes:** A fast, probabilistic approach focused on word distribution.

## 🛠️ Tech Stack
* **Transformers:** Hugging Face (AutoModel, Trainer API)
* **Deep Learning:** TensorFlow / Keras
* **Machine Learning:** Scikit-learn
* **Text Processing:** Hazm (Persian NLP toolkit), Shekar
* **Environment:** Kaggle/Colab with GPU acceleration

## 📈 Key Findings & Results
| Model | Accuracy | Recall | Precision | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **ParsBERT** | **89.5%** | **91.1%** | **88.4%** | **Top Performer.** Best at understanding context. |
| **Logistic Regression**| 85.8% | 85.0% | 86.5% | Strongest baseline; very efficient. |
| **Naive Bayes** | 85.1% | 91.0% | 80.8% | High recall; prone to false positives. |
| **LSTM** | 83.8% | 82.3% | 84.7% | Struggled to beat TF-IDF baselines. |

**Key Takeaway:** While Transformers provide the highest accuracy, **Logistic Regression** remains a highly competitive and lightweight alternative for this specific dataset size, outperforming the more complex LSTM.

## 📁 Repository Structure
* `snappfood-sentiment-analysis.ipynb`: The complete end-to-end pipeline, including data cleaning, class balancing, model training, and evaluation.

## ⚙️ How to Use
1.  **Clone the repo:**
    ```bash
    git clone [https://github.com/mahdisdg/snappfood-sentiment-analysis.git](https://github.com/mahdisdg/snappfood-sentiment-analysis.git)
    ```
2.  **Install dependencies:**
    ```bash
    pip install transformers hazm scikit-learn tensorflow
    ```
3.  **Dataset:** The notebook automatically fetches the dataset from Hugging Face (`PNLPhub/snappfood-sentiment-analysis`).
4.  **Run:** Open the `.ipynb` file in a GPU-enabled environment like Kaggle or Google Colab to see the training and comparison logic.

## 🧠 Skills Demonstrated
* **Cross-Architecture Evaluation:** Comparing Transformers vs. RNNs vs. Classical ML.
* **Persian NLP:** Handling RTL text, lemmatization, and Persian-specific tokenization.
* **Model Optimization:** Fine-tuning large language models (LLMs) and handling class imbalances.
