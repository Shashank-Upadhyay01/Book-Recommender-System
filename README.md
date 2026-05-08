# 📖 ReadRadar AI: Machine Learning Book Recommender

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)

**ReadRadar AI** is a behavior-driven recommendation engine that understands what readers love. By processing over 1 million data points, it uses high-dimensional vector math to suggest books based entirely on human rating patterns, served through a premium, glassmorphism-styled web interface.

### 🔴 **[Click Here to View the Live App!](https://book-recommender-system-shashank-upadhyay.streamlit.app/#read-radar-ai)**

---

## ✨ Features

* **Collaborative Filtering Engine:** Does not rely on basic metadata like genre or author. Instead, it calculates the geometric *Cosine Similarity* between books based on how actual users rated them.
* **Premium Dark UI:** Custom-built Streamlit frontend featuring keyframe animations, glassmorphism cards, and interactive hover physics to break away from the standard dashboard look.
* **Optimized Payload:** Engineered a robust data pipeline to filter out matrix sparsity (noise from low-interaction users) and drop duplicate entries, reducing the dataset payload to under 1MB for instant cloud rendering.
* **Global Search:** Instantly pulse the database of popular books to find 5 nearest-neighbor matches in real-time.

---

## 📸 Sneak Peek

### 1. The Home Feed
Home Page Screenshot<img width="2005" height="1277" alt="homepage" src="https://github.com/user-attachments/assets/4852e8f8-67be-4c05-80c8-9cf50e541de4" />
> *Dynamic trending feed featuring the highest-voted books in the database.*

### 2. The Recommendation Engine
Recommend Page Screenshot<img width="2005" height="1277" alt="recommend_page" src="https://github.com/user-attachments/assets/d27fc862-60fe-4713-962f-8707dc415bd3" />
> *Real-time inference using K-Nearest Neighbors to find visually similar books based on vector distances.*

### 3. System Architecture
About Page Screenshot<img width="2005" height="1277" alt="about_page" src="https://github.com/user-attachments/assets/220c1d88-2756-446a-8de9-b6c508388431" />
> *Detailed breakdown of the mathematical models and tech stack powering the platform.*

---

## 🧠 System Architecture & The Math

This system converts user ratings into a **high-dimensional matrix**, representing every book as a vector. 

1. **Sparsity Reduction:** Filtered out users who rated fewer than 200 books, and books with fewer than 50 ratings. This eliminates extreme noise and prevents the "Cold Start" problem.
2. **Vectorization:** Created a pivot table mapping explicit ratings to specific users.
3. **K-Nearest Neighbors (KNN):** Utilizes the `brute` algorithm to calculate the **Cosine Similarity** (the geometric angle) between the target book's vector and every other book. The shortest distances represent books enjoyed by the exact same group of readers.

---

## 💻 Run it Locally

Want to run the model on your own machine? 

**1. Clone the repository:**
```bash
git clone [https://github.com/yourusername/Book-Recommendation-Engine.git](https://github.com/yourusername/Book-Recommendation-Engine.git)
cd Book-Recommendation-Engine

**2. Install dependencies:**
pip install -r requirements.txt

3. Run the Streamlit server:
streamlit run app.py

📂 Dataset Acknowledgment
This project was trained using the Book-Crossing Dataset collected by Möbius, LINK HERE: https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset?resource=download.
