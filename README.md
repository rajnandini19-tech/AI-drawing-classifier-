# AI-drawing-classifier-
A hand drawn shape classifier built with python
# AI Drawing Classifier

A Python desktop app that learns to recognize hand-drawn shapes. Draw something on the canvas, label it, train a model on your own examples, and it predicts what you've drawn.

## Overview

This project builds a full pipeline: draw → save → preprocess → train → predict.

Instead of using a pre-made dataset like MNIST, the app lets you define your own classes (e.g. Triangle, Square, Circle) and create your own training data by drawing examples yourself, then trains a machine learning model on it.

## Features

- Live drawing canvas built with Tkinter
- Custom classes — define any 3 categories to classify
- Saved drawings are automatically organized into labeled folders
- Multiple ML models to choose from: Linear SVC, K-Nearest Neighbors, Logistic Regression, Decision Tree, Random Forest, Gaussian Naive Bayes
- Real-time prediction on new drawings
- Save/load trained models and full project state
- Adjustable brush size

## Tech Stack

- Python
- Tkinter (GUI)
- Pillow (image handling)
- OpenCV (image processing)
- NumPy
- scikit-learn (machine learning)
- Pickle (saving/loading)

## How It Works

1. On first launch, name your project and define three classes.
2. Draw a shape on the canvas, then click the matching class button to save it.
3. Once you have enough examples per class, click Train Model.
4. Draw something new and click Predict to see what the model thinks it is.
5. Use Change Model to switch between different ML algorithms.
6. Save your project to resume later — it reloads automatically next time you use the same project name.

## Getting Started

Install dependencies:
```
pip install pillow opencv-python numpy scikit-learn
```

Run the app:
```
python main.py
```

## Project Structure

```
ai-drawing-classifier/
├── main.py
└── <project_name>/
    ├── <class_1>/
    ├── <class_2>/
    ├── <class_3>/
    └── <project_name>_data.pickle
```

## What I Learned

- Building a full pipeline from GUI to trained model, not just training in a notebook
- Preprocessing raw drawings into a format a model can learn from
- Why training data quality and balance matters
- Debugging real issues: mismatched counters, coordinate bugs, and data pipeline errors that only show up at prediction time
- Comparing different ML algorithms on the same task

## Future Improvements

- Add data augmentation to improve accuracy with fewer drawings
- Show prediction confidence scores
- Deploy as a web app for a shareable demo

## Acknowledgements

Built while following NeuralNine's AI Drawing Classifier tutorial on YouTube, then debugged and extended.

[Rajnandini Chaudhari]
