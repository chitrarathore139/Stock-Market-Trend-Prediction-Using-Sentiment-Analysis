# 📈 Stock Market Trend Prediction Using Sentiment Analysis

An AI-powered Stock Market Trend Prediction system that combines Machine Learning, Deep Learning, and Natural Language Processing (NLP) to analyze market sentiment and predict stock trends in real time.

## 🚀 Features

* 📊 Real-time stock market data visualization
* 📰 Sentiment Analysis using news/headlines
* 🤖 Machine Learning & Deep Learning prediction models
* 📈 Interactive charts using Plotly
* 🌐 Streamlit-based responsive web application
* 🔍 Trend forecasting and market insights

## 🛠️ Technologies Used

* Python
* Streamlit
* TensorFlow / Keras
* Pandas & NumPy
* Scikit-learn
* Plotly & Matplotlib
* NLP Libraries (NLTK, TextBlob)
* yFinance API

---

# ⚙️ VS Code Extensions Recommended

## Python & AI

* Python
* Pylance
* Jupyter

## Git & GitHub

* GitLens
* GitHub Pull Requests and Issues

## Productivity

* Prettier
* Material Icon Theme
* Error Lens
* Code Runner

---

# 📥 Installation & Setup

## 1️⃣ Clone Repository

```bash
git clone YOUR_REPOSITORY_LINK
cd Stock-Market-Trend-Prediction-Using-Sentiment-Analysis
```

---

## 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

---

## 3️⃣ Activate Virtual Environment

### Windows

```bash
venv\Scripts\activate
```

### Mac/Linux

```bash
source venv/bin/activate
```

---

## 4️⃣ Install Required Packages

```bash
pip install -r requirements.txt
```

If requirements.txt is unavailable:

```bash
pip install streamlit pandas numpy matplotlib plotly scikit-learn tensorflow yfinance nltk textblob
```

---

# 📦 Download NLTK Data

Run Python:

```bash
python
```

Then execute:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
exit()
```

---

# ▶️ Run the Streamlit Application

```bash
streamlit run streamlit_app.py
```

Open in browser:

```bash
http://localhost:8501
```

---

# 🔧 Optional Warning Fix

Replace:

```python
use_container_width=True
```

With:

```python
width='stretch'
```

This removes Streamlit deprecation warnings.

---

# 📌 Future Improvements

* Live Twitter sentiment integration
* LSTM/Transformer-based forecasting
* Cloud deployment
* Portfolio recommendation system
* Advanced technical indicators

---

# 👨‍💻 Author

Chitra Rathore

