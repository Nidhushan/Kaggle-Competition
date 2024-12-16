# Kaggle-Competition
In this competition, participants are tasked with the Supervised-Fine Tuning(SFT) of Llama3-8B model to predict the correctness of answers to math questions. The goal is to assess whether the provided answer to each question is correct or not.

# Supervised Fine-Tuning of Meta-LLaMA-3.1-8B for Solution Verification to Math Problems

### Authors:  
- **Nidhushan Kanagaraja** - nk3755@nyu.edu  
- **Stavan Christian** - snc8114@nyu.edu  

---

## **Abstract**  
This project explores fine-tuning Large Language Models (LLMs) using the Low-Rank Adaptation (LoRA) technique. We fine-tuned Meta-LLaMA-3.1-8B to verify math solutions with explanations. This report includes model configurations, experimental methodologies, hyperparameter search, results, and future directions.

---

## **Table of Contents**  
1. [Introduction](#introduction)  
2. [Dataset](#dataset)  
3. [Model Information](#model-information)  
4. [Hyperparameter Experimentation](#hyperparameter-experimentation)  
5. [Results](#results)  
6. [Discussion](#discussion)  
7. [Conclusion](#conclusion)  
8. [Ethics Statement](#ethics-statement)  
9. [Acknowledgments](#acknowledgments)  
10. [References](#references)  

---

## **Introduction**  
Fine-tuning LLMs is computationally expensive due to the number of parameters involved. We leveraged LoRA to fine-tune Meta-LLaMA-3.1-8B efficiently. By freezing pre-trained weights and updating smaller weight matrices, we minimized computational overhead while maximizing task-specific performance.

---

## **Dataset**  
The dataset used was the "Math Question Answer Verification" dataset from Hugging Face:  
[Dataset Link](https://huggingface.co/datasets/ad6398/nyu-dl-teach-maths-comp)  

**Columns:**  
- `question`: The math question  
- `answer`: The correct answer  
- `solution`: Detailed explanation  
- `is_correct`: Correctness label (True/False)  

---

## **Model Information**  
We fine-tuned the Meta-LLaMA-3.1-8B using LoRA. The model was loaded with 4-bit quantization using Hugging Face libraries. Relevant layers such as `q_proj`, `k_proj`, `v_proj`, and others were modified using LoRA.

[Model Repository](https://huggingface.co/unsloth/Meta-Llama-3.1-8B-bnb-4bit)  

---

## **Hyperparameter Experimentation**  

### **Key Hyperparameters for LoRA**  
- **`r` (Rank of decomposition)**: Higher `r` retains more information but increases computational load.  
- **`lora_alpha`**: Scaling factor for LoRA layers.  
- **`lora_dropout`**: Regularization dropout rate.  
- **`use_rslora`**: Enables Rank-Stabilized LoRA (improves stability).  

### **Training Parameters**  
- `per_device_train_batch_size`  
- `gradient_accumulation_steps`  
- `warmup_steps`  
- `max_steps`  
- `learning_rate`  
- `weight_decay`  
- `lr_scheduler_type`  
- `max_grad_norm`

---

## **Results**  

| Model    | Accuracy | Loss  |
|----------|-----------|-------|
| Baseline | 0.60031  | 0.7296|
| v6       | 0.78703  | 0.413 |
| v7       | 0.78484  | 0.425 |
| v8       | 0.78425  | 0.438 |

---

## **Discussion**  
- **LoRA Efficiency**: Reduced memory and computational needs.  
- **Hyperparameter Impact**: Finding the right configuration improved results significantly.  
- **Prompt Design**: Structured, instructive prompts led to better performance.  
- **Challenges**: Limited resources constrained exploration of more advanced configurations.

---

## **Conclusion**  
Fine-tuning LLMs using LoRA proved to be computationally efficient. Future work should explore model generalization, task scalability, and transfer learning possibilities.

---

## **Ethics Statement**  
We emphasize transparency, fairness, and reproducibility by sharing our methods, code, and model weights.

---

## **Acknowledgments**  
Special thanks to Professor Gustavo Sandoval and teaching assistants for their support throughout the project.

---

## **References**  
1. [LoRA Parameter Documentation](https://docs.unsloth.ai/basics/lora-parameters-encyclopedia)  
2. [SFT Trainer Documentation](https://huggingface.co/docs/trl/main/en/sft_trainer#trl.SFTTrainer)  
3. [Training Arguments Documentation](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments)  
4. [GitHub Repository for Code](https://github.com/svenzer007/Supervised-fine-tuning-of-Meta-Llama-3.1-8B)  
5. [Kaggle Competition Page](https://www.kaggle.com/competitions/nyu-dl-fall-24-competition)
