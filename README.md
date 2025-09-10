# 📈 Wall Street LLM

_Bullish, Bearish, or Neutral? Benchmarking ML vs LLMs for Financial Sentiment._

You can read the complete article on [my Medium](https://medium.com/@falak.jain87/bullish-bearish-or-just-meh-fine-tuning-llms-to-beat-traditional-ml-at-financial-sentiment-e76f91ccc917).

---

## 🚀 Why Sentiment Matters

Ever wondered whether financial news is actually optimistic, pessimistic, or just plain neutral?  
This project compares **traditional ML models** (Random Forest, SVC, Logistic Regression) against **Large Language Models (LLMs)**, both base and fine-tuned, for **financial sentiment analysis**.

Using the [`zeroshot/twitter-financial-news-sentiment`](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment) dataset, we classify tweets and financial news into three categories:

- 🐂 **Bullish** (optimistic / positive sentiment)
- 🐻 **Bearish** (pessimistic / negative sentiment)
- 😐 **Neutral** (no clear market direction)

The result? **LLMs significantly outperform traditional ML** in capturing nuance, sarcasm, and financial jargon.

---

## 📊 Key Findings

- Traditional ML models achieved ~65–70% F1 scores.
- Base LLMs reached ~80–85% F1 scores out-of-the-box.
- Fine-tuned LLMs pushed performance above **90%**, especially on tricky cases like sarcasm, and shorthand.

---

## 🛠️ Tech Stack

- **Dataset:** Hugging Face `zeroshot/twitter-financial-news-sentiment`
- **Traditional ML:** scikit-learn (Random Forest, SVC, Logistic Regression)
- **Vectorization:** TF-IDF, embeddings
- **LLMs (base + fine-tuned):** Hugging Face Transformers (GPT 3.5, Claude, LLaMA, Qwen)
- **Evaluation:** Accuracy, F1-score, confusion matrix

---

## 📂 Repo Structure

```bash
wallstreet_llm/
│
├── scripts/                  # Python code and jupyter notebooks for experiments
│   ├── 01_data_prep.ipynb    # Data cleaning & preprocessing
│   ├── 02_ml_models.ipynb    # Random Forest, SVC, Logistic Regression
│   ├── 03_llm_baseline.ipynb # Zero-shot & base LLM evaluation
│   ├── 04_llm_finetune.ipynb # Fine-tuning on dataset
│   └── 05_results.ipynb      # Comparison, metrics, and visualizations
│
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── environment.yml           # YML file for environment set up
```

---

## Project Set-up (for Mac)

### **Clone the repository:**

Enter this in the terminal in your project folder:

`git clone https://github.com/falakjain98/wallstreet_llm.git`

### **Install Anaconda**

1. **Install Anaconda:**

- Download Anaconda from https://docs.anaconda.com/anaconda/install/mac-os/

2. **Set up the environment:**

- Open a **new** Terminal
- Navigate to the "project root directory"
- Create the environment: `conda env create -f environment.yml`
- Wait for a few minutes for all packages to be installed
- **Activate** the virtual environment using this command: `conda activate llms`
- You should see `(llms)` in your prompt, which indicates you've activated your new environment.

### **Create .env file**

- You will first have to create your API keys on the respective platforms before pasting here.
- The following services will be used in the course:
  - Open AI
  - Anthropic
  - HuggingFace
- Create a new file called `.env` in your project root directory.
- Then type your API keys into the file, replacing xxxx with your API key.

```
GOOGLE_API_KEY=xxxx
ANTHROPIC_API_KEY=xxxx
HF_TOKEN=xxxx
```

### **Use any IDE to run and edit scripts**

### **Showtime!!**

- Open Terminal
- Navigate to the "project root directory"

- Activate your environment with `conda activate llms`

- You should see (llms) in your prompt which is your sign that your environment is working. Now you can use the IDE of your choice to run and edit the scripts.

Please email me at falak.jain87@gmail.com in case you face any challenges.

## My Inspiration

I would like to give a shout out to [Edward Donner](https://edwarddonner.com) from whom I learned so much of what I know about LLM Engineering. His [GitHub](https://github.com/ed-donner) is full of interesting projects which you could fine useful too.
