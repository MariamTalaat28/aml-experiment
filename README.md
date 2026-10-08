
# Cloud-Based Adversarial Machine Learning for Network Intrusion Detection

This project investigates the vulnerability of deep learning-based network intrusion detection systems to adversarial attacks and evaluates different defense mechanisms against these attacks.

The practical experiment was conducted using the **CICIDS-2017** dataset on an **AWS EC2 CPU-based cloud instance**.

## Project Overview

The main objective of this project is to evaluate how adversarial machine learning techniques can affect a Deep Neural Network (DNN) used for binary network intrusion detection.

The project follows the following workflow:

**CICIDS-2017 Dataset**  
↓  
**Data Inspection & Preprocessing**  
↓  
**Binary Classification**  
↓  
**DNN Baseline Model**  
↓  
**Adversarial Attacks**  
↓  
**Attack Evaluation**  
↓  
**Defense Mechanisms**  
↓  
**Defense Evaluation**

## Dataset

The experiment uses the **CICIDS-2017** intrusion detection dataset.

The dataset contains network traffic generated from benign activities and different types of attacks.

For this experiment:

- **Task:** Binary intrusion detection
- **Classes:** `BENIGN` and `ATTACK`
- **Features:** 78 numerical features
- **Processed dataset:** 2,655,060 records
- **Balanced training data:** 729,126 records
- **Test data:** 530,453 records

The training dataset was balanced between the two classes before model training.

## Baseline Model

A Deep Neural Network (DNN) was developed for binary intrusion detection.

### Architecture

```text
Input Layer: 78 features
        ↓
Dense Layer: 128 neurons
        ↓
Dense Layer: 64 neurons
        ↓
Dense Layer: 32 neurons
        ↓
Output Layer: 1 neuron
```

### Training Configuration

| Parameter | Configuration |
|---|---|
| Model | DNN |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Batch Size | 256 |
| Epochs | 50 |
| Scaler | StandardScaler |
| Random Seed | 42 |

The baseline model achieved approximately **99.56% accuracy** on the test data.

## Adversarial Attacks

Four adversarial attack techniques were investigated.

### 1. FGSM

The **Fast Gradient Sign Method (FGSM)** was evaluated using three perturbation magnitudes:

```text
ε = 0.01
ε = 0.05
ε = 0.10
```

### 2. PGD

**Projected Gradient Descent (PGD)** was evaluated using:

```text
ε = 0.01
ε = 0.05
ε = 0.10
Steps = 10
```

### 3. C&W

A tabular adaptation of the **Carlini & Wagner (C&W)** attack was implemented.

```text
Steps = 100
Learning Rate = 0.005
C = 1.0
```

Due to its computational cost, the C&W experiment was evaluated using a fixed 1,000-sample subset.

### 4. DeepFool

A binary DeepFool-style adaptation was implemented.

```text
Steps = 100
Overshoot = 0.10
```

The experiment was also evaluated using the fixed 1,000-sample subset.

## Main Attack Results

| Attack | Configuration | Accuracy | Accuracy Drop |
|---|---|---:|---:|
| Clean Baseline | — | 99.56% | — |
| FGSM | ε = 0.10 | 35.52% | 64.04 pp |
| PGD | ε = 0.10, 10 steps | 38.94% | 60.62 pp |
| C&W | 100 steps | 27.51%* | 72.09 pp* |
| DeepFool | 100 steps | 73.09%* | 26.51 pp* |

\* C&W and DeepFool results were obtained using the 1,000-sample evaluation subset.

## Defense Mechanisms

Three defense approaches were evaluated.

### Adversarial Training

A multi-attack adversarial training approach was implemented using:

- FGSM
- PGD
- C&W
- DeepFool

The training process used a cyclic attack schedule during training.

### Defensive Distillation

Defensive distillation was evaluated as a model-based defense against adversarial perturbations.

### Feature Squeezing / Input Sanitization

Input sanitization was implemented by quantizing the input features before classification.

The final configuration used:

```text
256 quantization levels
Training percentile range: 1st–99th percentile
```

## Defense Evaluation

| Defense | Clean | FGSM | PGD | C&W | DeepFool |
|---|---:|---:|---:|---:|---:|
| No Defense | 99.56% | 35.52% | 38.94% | 27.51%* | 73.09%* |
| Adversarial Training | 96.30% | 92.50% | 91.90% | 86.40% | 86.70% |
| Defensive Distillation | 99.50% | 45.60% | 52.00% | 6.80% | 77.50% |
| Feature Squeezing | 99.80%† | 82.00% | 85.10% | 79.60% | 89.00% |

\* C&W and DeepFool undefended results are based on the 1,000-sample evaluation subset.

† Clean accuracy after feature squeezing.

The results show that defense performance depends on the specific attack and implementation configuration. Therefore, multiple attack families were evaluated rather than relying on a single adversarial attack.

## Cloud Environment

The practical experiment was conducted on **Amazon EC2**.

### Instance Configuration

```text
Instance: c7i-flex.large
CPU: 2 vCPU
Memory: 4 GiB RAM
GPU: None
Operating System: Ubuntu 24.04
Python: 3.12.3
```

The experiment was performed using CPU-based computation.

A Python virtual environment was used to isolate the project dependencies.

## Technologies and Libraries

The main technologies and libraries used in this project include:

- Python
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Adversarial Robustness Toolbox (ART)
- Matplotlib
- Seaborn
- Joblib
- AWS EC2

## Project Structure

```text
aml-experiment/
│
├── data/
│
├── results/
│
├── src/
│
├── .gitignore
│
└── README.md
```

The `src` directory contains the Python scripts used for:

- Dataset inspection
- Data preprocessing
- Machine learning data preparation
- DNN training
- FGSM attacks
- PGD attacks
- C&W attacks
- DeepFool attacks
- ART-based attack validation
- Adversarial training
- Defensive distillation
- Feature squeezing
- Input sanitization
- Defense evaluation
- Result processing

The `results` directory contains the generated experimental results, including attack evaluation results, defense evaluation results, ART validation results, and training histories.

## Reproducibility

The experiments use a fixed random seed:

```text
Random Seed = 42
```

The same preprocessing and feature-scaling procedures were used consistently across the experiments.

C&W and DeepFool experiments use a fixed 1,000-sample evaluation subset because of their higher computational requirements on the CPU-based environment.

## Results

The experiments demonstrate that the baseline DNN is highly vulnerable to adversarial manipulation.

At ε = 0.10, FGSM reduced the baseline accuracy from **99.56% to 35.52%**, while PGD reduced it to **38.94%**.

The C&W evaluation resulted in **27.51% accuracy**, while the DeepFool-style evaluation resulted in **73.09% accuracy** on the 1,000-sample evaluation subset.

The defense experiments showed that the evaluated defense mechanisms behaved differently depending on the attack type. Adversarial training substantially improved the evaluated FGSM, PGD, C&W, and DeepFool accuracies, while defensive distillation and feature squeezing showed different levels of effectiveness across the attack configurations.

These results highlight the importance of evaluating machine learning-based intrusion detection systems against multiple adversarial attack techniques rather than relying only on clean test accuracy.

## References

1. I. J. Goodfellow, J. Shlens, and C. Szegedy, "Explaining and Harnessing Adversarial Examples," *International Conference on Learning Representations (ICLR)*, 2015.

2. A. Madry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu, "Towards Deep Learning Models Resistant to Adversarial Attacks," *International Conference on Learning Representations (ICLR)*, 2018.

3. N. Carlini and D. Wagner, "Towards Evaluating the Robustness of Neural Networks," in *2017 IEEE Symposium on Security and Privacy*, San Jose, CA, USA, 2017, pp. 39–57.

4. S.-M. Moosavi-Dezfooli, A. Fawzi, and P. Frossard, "DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks," in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016, pp. 2574–2582.

5. M.-I. Nicolae et al., "Adversarial Robustness Toolbox v1.0.0," *arXiv preprint arXiv:1807.01069*, 2018.

6. N. Papernot, P. McDaniel, X. Wu, S. Jha, and A. Swami, "Distillation as a Defense to Adversarial Perturbations Against Deep Neural Networks," in *2016 IEEE Symposium on Security and Privacy*, San Jose, CA, USA, 2016, pp. 582–597.

7. I. Sharafaldin, A. H. Lashkari, and A. A. Ghorbani, "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization," in *Proceedings of the 4th International Conference on Information Systems Security and Privacy (ICISSP)*, 2018.

8. S. O. Amoran and A. O. Ibiyeye, "Adversarial Machine Learning Attacks on Cloud-Based AI Security Systems," *International Journal of Communication and Information Technology*, vol. 6, no. 2, pp. 70–78, 2025.

## Author

**Mariam Talaat**

Information Systems  
Arab Academy for Science, Technology and Maritime Transport

## Project Status

**Completed**

The project includes:

- CICIDS-2017 dataset preprocessing
- Binary intrusion detection
- Baseline DNN training
- FGSM attack
- PGD attack
- C&W attack
- DeepFool attack
- ART-based attack validation
- Multi-attack adversarial training
- Defensive distillation
- Feature squeezing
- Defense evaluation
- Experimental result analysis
