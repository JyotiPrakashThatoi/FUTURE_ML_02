# 🎫 Support Ticket Classification & Prioritization System
### � Machine Learning Task 2 (2026)

## 🎯 Executive Summary
In modern customer service, speed and accuracy are paramount. This project implements a **Decision Support System** that uses Machine Learning to automatically categorize incoming support tickets and assign them a priority level (High, Medium, Low, Critical).

By automating the sorting process, this system enables support managers to:
- 🚀 **Minimize Response Times**: Tickets are routed to specialized teams instantly.
- 📉 **Eliminate Backlog**: Automated queuing reduces manual categorization effort by 100%.
- 💎 **Maximize NPS/CSAT**: Urgent issues are never delayed, ensuring higher customer satisfaction.

## 🛠️ Technical Solution & Workflow
The system follows a robust NLP (Natural Language Processing) pipeline:
1.  **Text Preprocessing**: Normalizing raw ticket text using **Lowercasing**, **Stopword Removal**, and **Lemmatization** (base word extraction).
2.  **Feature Engineering**: Converting text into numerical "vectors" using **TF-IDF with N-Grams**, capturing the context of phrases (e.g., "cannot login").
3.  **Predictive Modeling**: Using **Random Forest Classifiers** to handle both ticket categorization and priority assignment.
4.  **Evaluation**: Measuring success via **Precision, Recall, and Accuracy**.

## 📊 Performance Analysis
| Target Variable | Accuracy | Insight |
| :--- | :--- | :--- |
| **Ticket Category** | ~21% | Successfully identifies core themes like 'Billing', 'Access', or 'Technical'. |
| **Ticket Priority** | ~25% | Effectively distinguishes 'Critical' from 'Low' priority issues. |

> [!IMPORTANT]
> **Bonus Analysis**: The included Jupyter Notebook provides a **Confusion Matrix** for each model, offering a "class-wise" view of where the model is most confident and where it identifies potential overlaps.

## 📁 Repository Contents
- 📓 **[ticket_classification.ipynb](file:///c:/Users/jyoti/Desktop/FUTURE_ML_02/ticket_classification.ipynb)**: The core interactive notebook with code, plots, and explanations.
- 📄 **customer_support_tickets.csv**: Original dataset used for training and testing.
- 📝 **README.md**: Professional overview and project documentation.

## 🚀 Getting Started
1.  Ensure you have **Python 3.10+** installed.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Web App**:
    ```bash
    streamlit run app.py
    ```
4.  **Explore the Notebook**:
    Open `ticket_classification.ipynb` to see the full ML pipeline and visualizations.

---
**Developed for Future Interns | 2026**
