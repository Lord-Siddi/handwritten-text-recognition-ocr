# Handwritten Text Recognition (HTR)

An **end-to-end handwritten text recognition system** that converts handwritten document images into digital text using deep learning and statistical language models.

The system combines a **CNN–BiLSTM–CTC neural architecture** for visual text recognition with **KenLM n-gram language models** for improved decoding accuracy.

This project demonstrates the complete OCR pipeline including:

- Handwritten line segmentation  
- Deep learning–based text recognition  
- Language model–assisted decoding  
- Evaluation using WER and CER metrics  
- Interactive demo interface using Gradio

---

# Project Architecture

The recognition pipeline works as follows:

```
Handwritten Document Image
           ↓
Line Segmentation (Projection Method)
           ↓
Image Preprocessing (Resize + Normalization)
           ↓
CNN Feature Extraction
           ↓
Bidirectional LSTM Sequence Modeling
           ↓
CTC Output Layer
           ↓
Decoding
   ├── Greedy CTC
   └── Beam Search + KenLM Language Model
           ↓
Recognized Digital Text
```

---

# Key Features

- End-to-end handwritten OCR pipeline  
- CNN + BiLSTM architecture for sequence modeling  
- CTC loss for alignment-free training  
- Beam search decoding with KenLM language models  
- Comparison between greedy decoding and language-model decoding  
- Evaluation using Word Error Rate (WER) and Character Error Rate (CER)  
- Gradio-based interactive demo for real-time handwriting recognition

---

# Model Architecture

The recognition model consists of:

### CNN Layers
Extract visual features from handwritten text images.

### Bidirectional LSTM Layers
Capture sequential dependencies between characters.

### CTC Loss
Enables training without explicit character segmentation.

Architecture summary:

```
Input Image (64×800)
        ↓
CNN Layers
        ↓
Feature Map → Sequence Representation
        ↓
BiLSTM Layers
        ↓
Dense + Softmax
        ↓
CTC Decoding
```

---

# Language Model Integration

To improve recognition accuracy, the project integrates **KenLM n-gram language models**.

The language model helps correct OCR errors by incorporating linguistic context.

Example:

```
Greedy Output:
hte cat sat on teh mat

Language Model Output:
the cat sat on the mat
```

Two language models were trained:

- **3-gram language model**
- **5-gram language model**

The decoding uses **beam search with CTC probabilities + LM scoring**.

---

# Dataset

The model is trained using the **IAM Handwriting Dataset**.

Dataset source:

- Teklia IAM-line dataset via HuggingFace

Dataset contains:

- Handwritten line images  
- Corresponding ground truth text labels

Dataset link:

https://huggingface.co/datasets/teklia/IAM-line

---

# Project Structure

```
HTR/
│
├── notebooks/
│   ├── model_training.ipynb
│   ├── language_model.ipynb
│   └── end_to_end_pipeline.ipynb
│
├── models/
│   ├── htr_model.keras
│   ├── vocab_list.pkl
│   ├── iam_lm_3gram.binary
│   └── iam_lm_5gram.binary
│
├── evaluation/
│   └── evaluation_results.csv
│
├── README.md
└── requirements.txt
```

---

# Model Weights

Due to GitHub file size limitations, trained model weights are **not stored in this repository**.

Download them from Google Drive:

📥 **Google Drive:**  
https://drive.google.com/drive/folders/1xtYanhtjtLU9lPsFq-e8Mp6izr7XAGgz?usp=sharing

After downloading, place the files inside:

```
models/
```

---

# Results

The performance was evaluated on the IAM test dataset.

| Decoder | Word Error Rate (WER) | Character Error Rate (CER) |
|--------|-----------------------|----------------------------|
| Greedy CTC | ~0.79 | ~0.65 |
| 3-gram LM | ~0.32 | ~0.13 |
| 5-gram LM | ~0.26 | ~0.10 |

The language model significantly improves recognition accuracy compared to greedy decoding.

---

# Evaluation Metrics

Two standard OCR metrics are used:

### Word Error Rate (WER)

Measures word-level errors.

```
WER = (Substitutions + Insertions + Deletions) / Total Words
```

### Character Error Rate (CER)

Measures character-level errors.

```
CER = (Substitutions + Insertions + Deletions) / Total Characters
```

---

# Demo Interface

The system includes a **Gradio-based interactive interface**.

Users can:

- Upload a handwritten image
- Automatically segment text lines
- View greedy vs language-model predictions
- Evaluate recognition accuracy

Example workflow:

```
Upload Image
     ↓
Line Segmentation
     ↓
OCR Prediction
     ↓
Language Model Correction
     ↓
Display Results
```

---

# Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- KenLM
- pyctcdecode
- Gradio
- HuggingFace Datasets

---

# Future Improvements

Possible improvements include:

- Transformer-based OCR models (e.g., TrOCR)
- Neural language models instead of n-grams
- Better page layout analysis
- Multi-language handwriting recognition
- Improved handwriting style generalization

---

# Author

Developed as part of a **machine learning project for handwritten text recognition**.
