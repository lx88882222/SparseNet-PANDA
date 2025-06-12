<!-- <div align="center">
  <img src="./img/logo_grid.png" alt="Logo" width="200">
</div> -->

# SparseNet: Exploring the Trade-off between Convolution and Attention in Gigapixel Object Detection

This repository contains the official implementation for the final project of the Machine Learning course (Spring 2025) at Tsinghua University. This project investigates the architectural trade-offs for efficient and accurate object detection on the gigapixel-level, highly sparse PANDA dataset.

This work is built upon two excellent open-source projects:
*   [**SparseFormer (liwenxi/SparseFormer)**](https://github.com/liwenxi/SparseFormer)
*   [**LSNet (THU-MIG/lsnet)**](https://github.com/THU-MIG/lsnet)

Our code starts from the [SparseFormer](https://github.com/liwenxi/SparseFormer) baseline and innovatively integrates the highly efficient **LS-Block** from [LSNet](https://github.com/THU-MIG/lsnet), leading to a valuable discovery about designing networks for this unique visual challenge.

---

## 🚀 Core Idea & Story

The central theme of this project is not just to achieve high performance, but to **understand *why* certain architectural choices succeed or fail** in the extreme environment of gigapixel object detection.

Our story unfolds in three acts:

1.  **Act I: Building a Strong ConvNet Baseline (`SparseNet`)**: We first question whether the complex self-attention mechanism in SparseFormer is truly necessary. We replace its core attention blocks with standard convolutional residual blocks, creating our baseline, `SparseNet`. This model surprisingly achieves a strong **0.70 AP50**, proving that a pure ConvNet is a viable contender.

2.  **Act II: The Quest for Ultimate Efficiency (`SparseNet-LS`)**: Inspired by the "See Large, Focus Small" principle of LSNet, we hypothesize that we can significantly reduce GFLOPs by swapping our local convolution blocks with the hyper-efficient LS-Blocks. This leads to the creation of the `SparseNet-LS` variant.

3.  **Act III: An Insightful Discovery**: The experiment yields a fascinating result. While `SparseNet-LS` successfully **lowers GFLOPs**, its **AP50 drops to 0.58**. This is not a failure, but a key insight: *for the sparse and high-variance nature of the PANDA dataset, the dynamic, adaptive modeling capability of self-attention (or a sufficiently complex ConvNet block) is more critical than the sheer computational efficiency of generalized lightweight modules like the LS-Block.*

---

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/lx88882222/SparseNet-PANDA.git
    cd SparseNet-PANDA
    ```

2.  **Create and activate a Conda environment:**
    ```bash
    conda create -n sparsenet python=3.8 -y
    conda activate sparsenet
    ```

3.  **Install dependencies:**
    This project is built upon MMDetection. Please install the necessary dependencies using the provided `requirements.txt` and by following the official MMDetection installation guide.
    ```bash
    pip install -r requirements.txt
    # You might need to install PyTorch and MMCV manually to match your CUDA version
    # pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    # pip install -U openmim
    # mim install mmcv-full
    # mim install mmdet
    ```

---

## 🔬 Usage

### Data Preparation

Please download the PANDA dataset from the [official website](https://www.gigavision.cn/data/news?nav=DataSet%20Panda&type=nav&t=1689145968317) and structure it as required by MMDetection.

### Inference with Our Best Model

You can easily reproduce the results of our best-performing model (`SparseNet`). The champion model weights and its corresponding configuration file are located in the `/checkpoints` directory.

To run distributed testing on **N** GPUs:
```bash
# Make sure you are in the project root directory
./tools/dist_test.sh \
    checkpoints/config.py \
    checkpoints/best_model.pth \
    N
```

### Training From Scratch

To train the models yourself, you can use the following commands for multi-GPU training.

*   **Train `SparseNet` (Our Baseline):**
    ```bash
    # This command trains the baseline model that achieved 0.70 AP50
    # Replace N with the number of GPUs you want to use.
    ./tools/dist_train.sh \
        [PATH_TO_YOUR_SPARSENET_CONFIG] \
        N \
        --work-dir ./work_dirs/my_sparsenet_experiment
    ```

*   **Train `SparseNet-LS` (Our Experiment):**
    ```bash
    # This command trains the experimental model with LS-Blocks
    # Replace N with the number of GPUs you want to use.
    ./tools/dist_train.sh \
        [PATH_TO_YOUR_SPARSENET_LS_CONFIG] \
        N \
        --work-dir ./work_dirs/my_sparsenet_ls_experiment
    ```

---

## 📈 Key Results

Our core findings are summarized in the table below, highlighting the trade-off between accuracy and efficiency.

| Model           | Core Local Module          | AP50 | GFLOPs (Relative) | Key Takeaway                               |
| --------------- | -------------------------- |:----:|:-----------------:|:-------------------------------------------|
| **`SparseNet`** | Standard Conv Residual Block | 0.70 | High              | Proves ConvNets are strong for this task.      |
| **`SparseNet-LS`**| LS-Block from LSNet        | 0.58 | **Low**           | Shows that efficiency alone is not enough. |

---

## 🎓 Conclusion & Contribution

This project provides a comprehensive study on vision architectures for gigapixel detection. Our main contributions are:
1.  We built and validated a strong pure-convolutional baseline, `SparseNet`, demonstrating its effectiveness.
2.  We conducted a novel experiment by integrating the `LS-Block` into our baseline, quantitatively revealing a critical accuracy-efficiency trade-off specific to HRW datasets.
3.  Our findings suggest that for sparse detection tasks, the architectural capacity for dynamic, fine-grained feature extraction is paramount, offering valuable insights for future network design in this domain.

### Acknowledgements

This work would not have been possible without the excellent codebases provided by the authors of [SparseFormer](https://github.com/liwenxi/SparseFormer) and [LSNet](https://github.com/THU-MIG/lsnet). We sincerely thank them for their open-source contributions to the community.



#### 📰 <a href="https://xxx" style="color: black; text-decoration: underline;text-decoration-style: dotted;">Paper</a>     :building_construction: <a href="https:/xxx" style="color: black; text-decoration: underline;text-decoration-style: dotted;">Model (via Google)</a>    :building_construction: <a href="https://pan.baidu.com/s/" style="color: black; text-decoration: underline;text-decoration-style: dotted;">Model (via Baidu)</a>    :card_file_box: <a href="https://www.gigavision.cn/data/news?nav=DataSet%20Panda&type=nav&t=1689145968317" style="color: black; text-decoration: underline;text-decoration-style: dotted;">Dataset</a>    :bricks: [Code](#usage)    :monocle_face: Video    :technologist: Demo    




### * The core code has been released. More docs will be updated in the future. Feel free to issue！




