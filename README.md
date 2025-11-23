# Visual-Question-Answering-VQA-project

This project implements a Visual Question Answering (VQA) system using the VQA v2.0 dataset. The system is designed to answer natural language questions related to images. The project is built with two main components: a **visual model** (ResNet50) for image feature extraction and a **text model** (BERT) for understanding and processing questions. These models are combined in the VQA model to predict answers.

---

## 🎯 General Purpose

The goal is to create a model that can answer questions about images by combining visual and textual information. The project uses a combination of computer vision and natural language processing to achieve this.

![VQA Demo Image](https://github.com/Taha-bouhafa1/Visual-Question-Answering-VQA-project/blob/main/vqa-image.png)

---

## 🧠 Architecture Overview

Below is the high-level architecture of the Visual Question Answering system, showing the flow of image and question processing:

![VQA Architecture](https://github.com/Taha-bouhafa1/Visual-Question-Answering-VQA-project/blob/main/Untitled-2025-07-02-1906.png)

---

## 🛠 Tools and Technologies Used

- **VQA v2.0 Dataset** ([Download VQA v2.0 Dataset Here](https://visualqa.org/download.html)) : The dataset includes images, questions, and answers. Only the training subset was used due to hardware limitations.
- **ResNet50**: A convolutional neural network used to extract image features.
- **BERT**: A transformer model (`bert-base-uncased`) used to process and extract features from textual questions.
- **PyTorch**: The deep learning framework used to train and deploy the models.
- **Matplotlib**: For visualizing results.
- **Kaggle**: The models were trained on Kaggle using two T4 GPUs.

---

## 🔁 Project Workflow

1. **Dataset Preprocessing**: 
   - Only the training data subset was used, with 20% for training, 5% for validation, and 5% for testing.
   - A vocabulary of the top 1000 most frequent answers was created.
   - Questions were tokenized using the BERT tokenizer.

2. **Model Components**:
   - **ResNet50** was trained to extract image features.
   - **BERT** (`BertForSequenceClassification`) was used for question feature extraction.

3. **VQA Model**: The features from the visual and text components were combined and fed into the VQA model, which was trained to predict answers.

4. **Training and Evaluation**: The models were trained using the PyTorch framework, with evaluation conducted on a held-out test set.

---

## 📊 Evaluation Metrics

- **Test Accuracy**:
  - **ViLBERT**: 71.79%
  - **VisualBERT**: 70.80%
  - **Our Model**: 43.46%

Our model's test accuracy is lower than state-of-the-art models like ViLBERT and VisualBERT due to:

- 🧠 Limited hardware resources (Kaggle T4 GPUs)
- ⏳ Smaller dataset used (only 20% of training data)
- 🔄 Reduced training epochs

Despite these constraints, the model lays a strong foundation for further improvement.

---

## 📁 Repository Contents

- **Notebooks**:
  - `bert_training.ipynb`: BERT training notebook.
  - `resnet_training.ipynb`: ResNet training (ResNet34 and ResNet50, with ResNet50 final).
  - `vqa_model_training.ipynb`: VQA model training notebook.
  - `demo.ipynb`: Demo notebook for testing the VQA model.

- **Output Files**:
  - `annotations_with_majority_answers.json`
  - `answer_vocab.json`
  - `test_prediction.json`
  - `tokenized-questions_with_ids.json`

- **Models**:
  - `bert_model_updated.pth`
  - `best-resnet34.pth`
  - `best_resnet50.pth`
  - `best_resnet50_v1.pth`
  - `best_vqa_model.pth`

- **Reports**: Detailed project report (PDF).
- **Presentations**: Slides explaining the project approach and results (PDF).

---

## 🚀 How to Run the Project

```bash
# Clone the repository
git clone https://github.com/Taha-bouhafa1/Visual-Question-Answering-VQA-project.git

# Navigate to the project directory
cd Visual-Question-Answering-VQA-project

# (Optional) Create and activate a virtual environment
python -m venv vqa-env
source vqa-env/bin/activate  # On Windows use: vqa-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the notebooks
jupyter notebook
```
##  Citation
```bash
@misc{bouhafa2025vqa,
  author       = {Taha Bouhafa and Loubaba Lhlaibi Lmalki},
  title        = {Visual Question Answering using ResNet50 and BERT: A Multimodal Deep Learning Approach},
  year         = {2025},
  institution  = {University Abdelmalek Essaadi, National School of Applied Sciences of Tétouan},
  supervisor   = {Prof. Belcaid Anass},
  howpublished = {\url{https://github.com/Taha-bouhafa1/Visual-Question-Answering-VQA-project}},
  note         = {Deep Learning Project}
}
```
## License

This project is licensed under the **MIT License**.  
See the **[LICENSE](LICENSE)** file for more information.


