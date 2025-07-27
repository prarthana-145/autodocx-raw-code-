# AutoDocX: AI-Powered Receipt Data Extraction

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://www.python.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-yellow)](https://huggingface.co/transformers/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)

AutoDocX is an end-to-end solution for extracting structured information from receipt images using a fine-tuned **LayoutLMv3** model. This project leverages the power of multimodal AI, understanding both the text and the visual layout of a document to achieve high-accuracy data extraction.

You can find another subpart of this project: Document Classification using Resnet with FastAPI here : [Link](https://github.com/prarthana-145/document_classificaton)
---

## ✨ Key Features

-   **Multimodal Understanding:** Utilizes **LayoutLMv3**, which processes text, layout, and visual information simultaneously for superior accuracy.
-   **End-to-End Pipeline:** Covers the entire process from data preparation and model training to inference on new, unseen images.
-   **High-Accuracy OCR:** Employs **Tesseract** for initial text extraction, providing the raw data for the model.
-   **Hyperparameter Optimization:** Integrates **Optuna** to automatically find the most effective training parameters, maximizing model performance.
-   **Advanced Post-Processing:** Implements intelligent, rule-based heuristics to clean and structure the model's raw output, correctly identifying line items and footer details.
-   **Line Confidence Scoring:** A unique feature that calculates a confidence score for each detected line of text, allowing the system to filter out OCR noise and irrelevant information by applying a threshold.

---

## ⚙️ Project Workflow

The project follows a systematic pipeline to train the model and perform inference:

1.  **Data Preparation:** The **CORD dataset** (containing receipt images and JSON annotations) is loaded. The `LayoutLMv3Processor` tokenizes text, normalizes bounding box coordinates, and aligns all inputs for the model.
2.  **Model Fine-Tuning:** A pre-trained `microsoft/layoutlmv3-base` model is fine-tuned on the CORD dataset. The model learns to classify each word/token into predefined categories (e.g., `menu.nm`, `total.total_price`).
3.  **Inference on New Images:**
    -   An unseen receipt image is processed with **Tesseract OCR** to extract words and their bounding boxes.
    -   The fine-tuned **LayoutLMv3 model** predicts a label for each word.
    -   **Post-processing rules** are applied to refine these predictions, group words into logical entities (like multi-word item names), and format the final output.

---

## 🛠️ Tech Stack

-   **Core Model:** `LayoutLMv3` (from Hugging Face Transformers)
-   **Framework:** `PyTorch`
-   **OCR Engine:** `Tesseract`
-   **Data Libraries:** `Hugging Face Datasets`, `Pandas`
-   **Hyperparameter Tuning:** `Optuna`
-   **Environment:** `Jupyter Notebook`, `Python`

---

## 🚀 Getting Started

### Prerequisites

-   Python 3.8+
-   Tesseract OCR engine installed on your system.
-   An environment with PyTorch and other required packages.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/autodocx.git](https://github.com/your-username/autodocx.git)
    cd autodocx
    ```

2.  **Install Python packages:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You will need to create a `requirements.txt` file containing the necessary packages like `transformers`, `torch`, `pytesseract`, `datasets`, `optuna`, etc.)*

### Training the Model

1.  **Download the Dataset:** Place the CORD dataset in the appropriate directory as referenced in the notebook.
2.  **Configure Paths:** Open the `autodocx.ipynb` notebook and update the file paths in the configuration cells to match your local setup.
3.  **Run the Notebook:** Execute the cells in sequence to preprocess the data, run hyperparameter tuning (optional), and train the final model. The fine-tuned model will be saved to the `final_layoutlmv3_model` directory.

### Running Inference

1.  Ensure your fine-tuned model is saved.
2.  In the **Inference** section of the notebook, update the `test_image_path` variable to point to your receipt image.
3.  Run the inference cells to see the extracted data in a structured table and JSON format.

---


