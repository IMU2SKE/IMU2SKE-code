# Gait Freezing Prediction

This project focuses on predicting gait freezing in individuals using various deep learning models. It includes scripts for data preprocessing, model training, validation, and testing.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/IMU2SKE/IMU2SKE-code.git
   cd IMU2SKE-code
   ```

2. **Install dependencies:**
   It is recommended to use a virtual environment.
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

   Install the required packages using pip:
   ```bash
   pip install -r requirements.txt
   ```

## Data Preprocessing

1. **Install ONNX and ONNX Runtime:**
   If you have a GPU machine, run:
   ```bash
   pip install onnx onnxruntime-gpu
   ```
   If you only have a CPU machine, run:
   ```bash
   pip install onnx onnxruntime
   ```

2. **Download weights:**
   ```bash
   huggingface-cli download tzhhhh/sv4pdd-dwpose --local-dir ckpts
   ```
   The `dwpose` code is borrowed from [dwpose-onnx](https://github.com/IDEA-Research/DWPose).

## Usage

This project uses shell scripts to run the training

### Training

To train a model, you can use the `train.sh` script. You can customize the model, learning rate, epochs, and other parameters within the script.

```bash
bash Model/imu_train.sh
```

