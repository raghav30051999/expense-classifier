# expense-classifier
Every time you swipe a credit card, a transaction is recorded — but it does not always come with a clear label like "Grocery" or "Travel". This project builds a Machine Learning model that automatically reads transaction details and classifies it into one of **14 spending categories** such as `gas_transport`, `shopping_pos`, or `health_fitness`.

# Expense Classifier — Predicting Transaction Categories Using Machine Learning

## What Does This Project Do?
Every time you swipe a credit card, a transaction is recorded — but it does not always come with a clear label like "Grocery" or "Travel". This project builds a Machine Learning model that automatically reads transaction details and classifies it into one of **14 spending categories** such as `gas_transport`, `shopping_pos`, or `health_fitness`.

Think of it as a smart assistant that looks at your transaction and says — *"This looks like a Food & Dining expense."*

---

## Dataset
**Source:** Kaggle Credit Card Transactions Dataset  
**Rows used:** 50,000 transactions  
**Input features:** Merchant name, Transaction amount, Gender, Job, City population  
**Output (what we predict):** 14 expense categories  
**Class balance:** The largest category has only 3.3x more rows than the smallest — healthy enough to train without any special balancing technique

---

## Key Challenge
All merchant names in this dataset are anonymized (e.g., `fraud_Weber and Sons`),
carrying no semantic business meaning unlike real names such as `Shell`, `Amazon`,
or `Starbucks`.

Three classical ML methods — Naive Bayes, SVM, and Logistic Regression — were
evaluated first. All three produced Macro F1 ≈ 0.11 due to heavy feature overlap
in Tier 3 categories, making them insufficient for this task.

BERT was then used, taking all available features — merchant name, amount, gender,
job, and city population — as a single combined text input. This collective feature
analysis produced Macro F1 ≈ 0.99 on a row-level split, where training merchants
reappeared in the test set.

The key limitation: when tested on completely unseen merchants (merchant-group split),
F1 dropped to ≈ 0.09–0.11. This confirms that prediction accuracy for unseen merchants
is directly dependent on data quality. A dataset with real merchant names and
transaction descriptions would allow the model to generalize reliably to new merchants.

---

## What is Macro F1 Score? (For Non-Technical Readers)
Accuracy tells you *how often* the model is correct overall. But if the model always predicts the most common category, accuracy looks good even though the model is useless.

**Macro F1** is a stricter measure — it checks how well the model performs on *every single category*, and averages them equally. A score of **1.0 is perfect**, and **0.0 means the model is completely wrong**. So our range of Macro F1 score is 0.0 to 1.0.

---

## Data Analysis — Three Difficulty Tiers
Before training any model, we analysed transaction amounts per category and found a clear pattern:

| Tier              | Categories                                                        | Median Amount             | Difficulty                      |
|-------------------|-----------------------------------------------------------------  |---------------------------|---------------------------------|
| Tier 1            | grocery_pos, gas_transport, grocery_net, entertainment            | $52–$105                  | Easy — well separated           |
| Tier 2            | home, kids_pets, food_dining, health_fitness, personal_care       | $32–$48                   | Moderate — needs extra features |
| Tier 3            | misc_pos, misc_net, shopping_net, shopping_pos, travel            | $6–$13                    | Hard — heavy amount overlap     |

**Tier 3 is the core challenge** — five categories all have transaction amounts between $6–$13, making them nearly impossible to separate using numbers alone. This is why a deep learning model (BERT) was needed.

---

## BERT Training
BERT (Bidirectional Encoder Representations from Transformers) is a language model pre-trained by Google on 3.3 billion words. Instead of training from scratch — which would require weeks of computation — we fine-tuned this pre-trained model on our 50,000 transactions. Fine-tuning means we took BERT's existing language understanding and adapted it specifically to recognize spending patterns from transaction text. All available features — merchant name, amount, gender, job, and city population — were combined into a single sentence and fed into BERT as input. Training was run for 2 epochs on a Kaggle GPU (T4 x2), taking approximately 20–30 minutes. This approach is significantly more powerful than classical ML because BERT reads the entire transaction as context rather than treating each feature as an isolated number.


## Experiments and Results
The test data of 50K rows was splitted into 40K rows for training and 10K rows for validation and test.
Note : The merchant names present in training data and test data are same, but remaining features may differ. 

| Experiment                    | Evaluation Setup                  | Macro F1      | What This Means                                    |
|-------------------------------|-----------------------------------|---------------|----------------------------------------------------|
| Classical ML (LR, NB, SVM)    | Tabular features, row-level split | ~0.11         | Tier 3 overlap makes tabular features insufficient |
| BERT                          | Row-level split                   | ~0.99         | high — merchant identity memorized                 |

The accuracy for predicting a transaction done through a known merchant is very high. 

But the key issue is what if a new transaction where the merchant name is different from the training data.

## Key issue
Prediction of transaction accurately, if merchant name was unseen during the training has became a key issue.
The test data of 50K rows was splitted into 40K rows for training and 10K rows for validation and test.
Note : The merchant names present in training data and test data are different, the test data may contain new merchant names which are completely unseen during the training.  

| Experiment                    | Evaluation Setup                  | Macro F1      | What This Means                                    |
|-------------------------------|-----------------------------------|---------------|----------------------------------------------------|
| BERT                          | Merchant-group split              | ~0.09–0.11    | True generalization — poor on unseen merchants     |

So BERT is working 
When we first trained BERT using a standard random split, the model achieved a Macro F1 of **0.99** — which looked excellent. But on closer inspection, the same merchant names were appearing in both training and test data.

**The fix:** We used a technique called `GroupShuffleSplit` which ensures that any merchant appearing in training data is completely absent from the test data. This tests whether the model truly learned spending patterns — not just merchant names. 

## Why Did BERT Struggle on Unseen Merchants?
BERT is a powerful language model that understands the *meaning* of words. It works brilliantly when text carries real meaning — for example, the word `Starbucks` signals coffee, `Uber` signals transport, `Walmart` signals grocery.

But in this dataset, merchant names like `fraud_Weber and Sons` carry **no real-world meaning**. BERT has nothing meaningful to read — so it performs no better than classical methods on unseen merchants.


## Conclusion
- For **recurring known merchants** — the model performs strongly (Macro F1 ≈ 0.99)
- For **completely new unseen merchants** — performance drops because anonymized names carry no business meaning
- The bottleneck is **data quality**, not model architecture
- With real merchant names, MCC codes, or transaction descriptions, BERT would generalize reliably.
- Real banking systems like Razorpay, PhonePe, HDFC all have mostly recurring merchants. A model that works perfectly on known merchants and gracefully acknowledges unseen merchant limitations is production-realistic. 

## Production Recommendation
| Scenario                  | Recommended Approach                                         |
|---------------------------|--------------------------------------------------------------|
| Known recurring merchants | Merchant-history lookup → high confidence prediction         |
| Unseen merchants          | Fallback model using amount, MCC codes, timestamps, location |
| Ideal dataset             | Real merchant names + transaction descriptions + MCC codes   |


## Tech Stack

| Component         | Tool                                  |
|-------------------|---------------------------------------|
| Language          | Python 3.10+                          |
| Environment       | Kaggle Notebooks (GPU T4 x2)          |
| Classical ML      | scikit-learn                          |
| Deep Learning     | HuggingFace Transformers, PyTorch     |
| Model             | bert-base-uncased (110M parameters)   |
| Split Strategy    | GroupShuffleSplit by merchant         |
| Primary Metric    | Macro F1                              |

