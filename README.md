# Iris Classifier (Decision Tree)

## Overview
This project demonstrates an end-to-end machine learning workflow using **scikit-learn**.  
It trains a **Decision Tree Classifier** on the classic [Iris dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-dataset), evaluates its performance,  
and saves a **confusion matrix plot** to `outputs/confusion_matrix.png`.

The project is part of the **Digital Marketing Mastery Module** as a hands-on introduction to ML pipelines.

---

## Quick start

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/moaiyadi/iris-classifier.git
cd iris-classifier

# create virtual environment
python -m venv irisVenv        #here irisVenv is the name of the virtual enviroment and it can be anyhting you wish
source irisVenv/bin/activate   # On Windows use: irisVenv\Scripts\activate

# install dependencies
pip install -r requirements.txt

# run training script
python src/train.py --test-size 0.2 --random-state 42

--

### Arguments

The training script accepts two optional arguments:

--test-size (float, default: 0.2)
Proportion of the dataset used for testing.
Example: --test-size 0.3

--random-state (int, default: 42)
Random seed for reproducibility.
Example: --random-state 123

--

## Outputs

Running the script will:

Print the model accuracy to the console.
Example:

Accuracy: 0.9667


Save the confusion matrix plot to:

outputs/confusion_matrix.png

--

## Project Structure
iris-classifier/
│
├── src/
│   └── train.py          # Main training script
│
├── outputs/              # Folder for generated outputs (auto-created)
│   └── confusion_matrix.png
│
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
└── LICENSE               # MIT license


--

## Requirements

Python 3.8+
scikit-learn
matplotlib
seaborn

Install them all with:
pip install -r requirements.txt

--

## License
This project is for educational purposes as part of the Digital Marketing Mastery Module.