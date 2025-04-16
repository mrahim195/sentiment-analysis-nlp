# 🧠 Sentiment Analysis on Social Media Posts

This is a Natural Language Processing (NLP) project developed during my internship at **CodexCue**, where I analyzed social media posts (specifically tweets) and classified their sentiments as **Positive**, **Negative**, or **Neutral** using machine learning.


## 📌 Project Overview

In this project, I used a real-world dataset containing airline-related tweets. The main goal was to apply **text preprocessing**, **vectorization**, and **classification algorithms** to detect the sentiment behind each tweet.


## 🧰 Tech Stack

- **Language:** Python
- **Libraries Used:** 
  - `pandas`
  - `scikit-learn`
  - `nltk`
  - `matplotlib`
  - `seaborn`

- **IDE:** Visual Studio Code (VS Code)


## 🚀 How to Run This Project

Follow these steps to set up and run the project locally on your machine:

### 1. Clone the Repository

``` bash
git clone https://github.com/mrahim195/sentiment-analysis-nlp.git
cd sentiment-analysis-nlp 
```

### 2. Create and Activate a Virtual Environment

# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate

### 3. Install Required Libraries

pip install pandas scikit-learn nltk matplotlib seaborn

### 4. Download NLTK Stopwords

# Add this in your Python script or run it separately
import nltk
nltk.download('stopwords')

### 5. Add the Dataset

Download the dataset from Kaggle:
🔗 Twitter US Airline Sentiment Dataset (https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment?resource=download)

Place Tweets.csv in the same folder as your script.

### 6. Run the Script
Make sure you're in the virtual environment and then run:

python main.py

### 📊 Sample Output

✅ Accuracy: 76%

🧮 Confusion Matrix

📈 Precision / Recall / F1 Score for each sentiment class

Output includes a full model evaluation with results printed in the terminal.

### 📇 Contact Me
Created by Muhammad Rahim Shahid
📫 Connect with me on LinkedIn: https://www.linkedin.com/in/muhammad-rahim-shahid-b04986268/
                       Gmail: mr5270229@gmail.com
