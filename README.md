# Visual-Question-Answering-VQA-project

This project implements a Visual Question Answering (VQA) system using the VQA v2.0 dataset. The system is designed to answer natural language questions related to images. The project is built with two main components: a **visual model** (ResNet50) for image feature extraction and a **text model** (BERT) for understanding and processing questions. These models are combined in the VQA model to predict answers.

## General Purpose
The goal is to create a model that can answer questions about images by combining visual and textual information. The project uses a combination of computer vision and natural language processing to achieve this.

## Tools and Technologies Used
- **VQA v2.0 Dataset**: The dataset includes images, questions, and answers. Only the training subset was used due to hardware limitations.
- **ResNet50**: A convolutional neural network used to extract image features.
- **BERT**: A transformer model (`bert-base-uncased`) used to process and extract features from textual questions.
- **PyTorch**: The deep learning framework used to train and deploy the models.
- **Matplotlib**: For visualizing results.
- **Kaggle**: The models were trained on Kaggle using two T4 GPUs.

## Project Workflow
1. **Dataset Preprocessing**: 
   - Only the training data subset was used, with 20% for training, 5% for validation, and 5% for testing.
   - A vocabulary of the top 1000 most frequent answers was created.
   - Questions were tokenized using BERT tokenizer.
2. **Model Components**:
   - **ResNet50** was trained to extract image features.
   - **BERT** (`BertForSequenceClassification`) was used for question feature extraction.
3. **VQA Model**: The features from the visual and text components were combined and fed into the VQA model, which was trained to predict answers.
4. **Training and Evaluation**: The models were trained using the PyTorch framework, with evaluation conducted on a held-out test set.

## Evaluation Metrics
- **Test Accuracy**:
  - **ViLBERT**: 71.79%
  - **VisualBERT**: 70.80%
  - **Our Model**: 43.46%

While our model's test accuracy is lower compared to state-of-the-art models like ViLBERT and VisualBERT, the performance is limited by several factors:
- **Hardware limitations**: Training was performed on Kaggle using two T4 GPUs, which impacted the ability to process the large dataset.
- **Dataset size**: Only a subset of the training data (20%) was used for training due to the large size of the VQA v2.0 dataset.
- **Reduced training time**: Due to hardware constraints, the model could not be trained for as many epochs as would be ideal.

Despite these challenges, the model provides a solid foundation and can be further improved with better hardware, additional training data, and more training time.

## Repository Contents
- **Notebooks**:
  - `bert_training.ipynb`: BERT training notebook.
  - `resnet_training.ipynb`: ResNet training (ResNet34 and ResNet50, with ResNet50 final).
  - `vqa_model_training.ipynb`: VQA model training notebook.
  - `demo.ipynb`: Demo notebook for testing the VQA model.
- **Output Files**:
  - `annotations_with_majority_answers.json`: Annotations with majority answers.
  - `answer_vocab.json`: Vocabulary of top 1000 most frequent answers.
  - `test_prediction.json`: Model test predictions.
  - `tokenized-questions_with_ids.json`: Tokenized questions with IDs.
- **Models**:
  - `bert_model_updated.pth`: Updated BERT model weights.
  - `best-resnet34.pth`: Best weights for ResNet34.
  - `best_resnet50.pth`: Best weights for ResNet50.
  - `best_resnet50_v1.pth`: Alternative ResNet50 model weights.
  - `best_vqa_model.pth`: Best trained VQA model weights.
- **Reports**: Detailed project report (PDF).
- **Presentations**: Slides explaining the project approach and results (PDF).

## Authors
- **Taha Bouhafa**
- **Loubaba Lhlaibi Lmalki**

## Supervisor
- **Prof. Belcaid Anass**

## Institution
- **University Abdelmalek Essaadi, National School of Applied Sciences of Tétouan**
