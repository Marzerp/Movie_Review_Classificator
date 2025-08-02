# Movie_Review_Classificator
### Classifies popular movie reviews into positive or negative 


---

## Overview
This proyect implements several **supervised learning models** for **binary sentiment classification** of movie reviews (positive or negative), using a publicly available dataset from Stanford University.

---

## Author

This project was created by Araceli Romero, a student of Information and Communication Technologies (TICs) at [UNAM](https://www.unam.mx/), Mexico.

Feel free to reach out with questions, feedback, or collaboration ideas to araceliromerozerpa@gmail.com

---

## License

This projects is under [MIT License](https://github.com/Marzerp/Movie_Review_Classificator/blob/main/LICENSE).

---

## Introduction 

Sentiment analysis of movie reviews has become an essential task in Natural Language Processing (NLP), with applications ranging from recommendation systems to consumer insight analysis. In this project, we explore the automatic classification of movie reviews into positive or negative sentiments using supervised machine learning models.

---

## Aim

The goal is to compare the performance of several well-established algorithms—including Multinomial Naive Bayes, Logistic Regression, Support Vector Machines (SVM), and eXtreme Gradient Boosting (XGBoost)—using a standard movie review dataset from Stanford. Each model is trained on TF-IDF representations of the reviews and evaluated based on accuracy, precision, recall, and F1-score.

Through careful preprocessing, feature extraction, and hyperparameter tuning, this project demonstrates how traditional models can still offer strong performance for text classification tasks—particularly when combined with effective representation techniques.

---

## Instalation 

### Requirements
- Python 3.10 

### Setup Instructions  
1. Clone the repository
   ```bash
   git clone https://github.com/Marzerp/Movie_Review_Classificator.git
   ```
   
2. Install requirements.txt
   ```bash
   pip install -r requirements.txt
   ```

3. Run the proyect
   ```bash
   python3 analisissentimientos_imdb.py
   ```
   
---

## Technologies

- Python3
- Scikit-learn 
- Pandas
- Numpy
- Matplotlib
- Nltk

---

## Conclusions

This project evaluated four supervised learning models—Multinomial Naive Bayes, Logistic Regression, Support Vector Machines (SVM with RBF kernel), and XGBoost—for the task of movie review sentiment classification. All models were trained and optimized using TF-IDF representations with n-grams, cross-validation, and hyperparameter tuning via GridSearch.

Among all models, SVM with RBF kernel achieved the best performance, reaching 89% accuracy with balanced precision and recall. Its ability to capture nonlinear decision boundaries proved effective for handling the complex patterns in review language.

Logistic Regression also performed competitively, benefiting from the linear nature of TF-IDF vectors. Despite its simplicity, Naive Bayes delivered solid results, especially given the independence assumptions that align well with bag-of-words representations. XGBoost, although powerful in other domains, underperformed here—likely due to the sparse, high-dimensional nature of the feature space, which favors linear models.

One key limitation of this study is the reliance on bag-of-words and TF-IDF, which do not capture semantics or word context. Additionally, the feature space was limited to 10,000 terms, potentially restricting the models' ability to learn deeper patterns.

Future improvements could involve the use of word embeddings (e.g., Word2Vec, GloVe) or contextual language models like BERT. Model ensembles and more efficient hyperparameter search techniques could also enhance performance. Finally, evaluating model behavior on different review lengths could provide deeper insights into their generalization capacity.

Despite these limitations, the results show that even relatively simple models—when properly preprocessed and tuned—can achieve strong performance in sentiment analysis tasks. In this study, SVM stood out as the most effective approach.






















