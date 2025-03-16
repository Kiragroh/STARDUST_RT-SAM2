# 🌟 STARDUST-MedSAM2 🌟

![STARDUST-MedSam-MG Banner](assets/bannerCollage.png)

## 🔍 Project Overview

STARDUST (Segmentation of Tumors via AI-powered Radiotherapy Database Utilization using Segment Anything Model 2 Technology) is an advanced medical imaging segmentation framework designed to enhance tumor segmentation in radiotherapy planning. Building upon Meta's state-of-the-art SAM2 foundation model (released July 31, 2024), STARDUST-MedSAM2 extends these capabilities to medical imaging contexts with specialized adaptation for 3D medical data.

### 🏥 Why Does This Matter?
Medical imaging plays a crucial role in cancer diagnosis and treatment, with tumor segmentation being a critical task in radiotherapy (RT). Traditionally, radiation oncologists manually outline tumors on CT and MRI scans, a time-consuming and highly variable process. While AI has already improved organ segmentation, precise tumor segmentation remains a major challenge. The STARDUST project leverages cutting-edge AI to significantly enhance the speed and consistency of tumor delineation, making RT planning more efficient and precise.

Recent advances in artificial intelligence (AI) have opened new opportunities for medical image segmentation, particularly with interactive medical image segmentation (IMIS). IMIS enhances accuracy by integrating iterative feedback from medical professionals. However, most methods struggle with the limited availability of 3D medical data, making generalization difficult. The Segment Anything Model (SAM) was initially developed for 2D segmentation, requiring extensive manual slice-by-slice annotations when applied to 3D medical imaging. The next-generation SAM2, trained on videos, introduces a paradigm shift by enabling full annotation propagation from a single 2D slice to an entire 3D medical volume. 

STARDUST-MedSAM2 builds upon this capability, leveraging a decade’s worth of radiotherapy treatment data—including 12,000 treatment courses with CT and MRI scans—to refine and optimize tumor segmentation for clinical applications. Unlike traditional medical imaging models that require training on limited datasets, SAM2's foundation on a billion+ annotations allows for superior adaptability. Our goal is to harness these advancements to improve radiotherapy planning, adaptive therapy, and follow-up care.

<div align="center">
  <img src="assests/principle_diagram.png" alt="Principle Diagram of STARDUST-MedSAM2" width="800"/>
</div>

> *This implementation adapts and extends the approach from [MedSAM2](https://github.com/bowang-lab/MedSAM/tree/MedSAM2), optimizing it for radiotherapy applications with a specific focus on gross tumor volume (GTV) segmentation.*

## 🚀 Features

- 🏥 **Medical Image Processing**: Specialized conversion from DICOM to NPZ format for optimal model compatibility
- 🧠 **Advanced Segmentation Methods**: 
  - ✅ SAM2 2D segmentation with middle-slice propagation
  - ✅ MedSAM2 2D segmentation with middle-slice propagation
  - ✅ SAM2 3D propagation-based segmentation
  - ✅ MedSAM2 3D propagation-based segmentation (in testing)
  - 🔬 Point prompt segmentation methods (in testing)
- 🎯 **Interactive GUI**: User-friendly interface for prompt-based segmentation
- 📊 **Comprehensive Evaluation**: Metrics computation for segmentation quality assessment
- 🔄 **Fine-tuning Capabilities**: Adapt pre-trained models to specific medical datasets

> ⚠️ **Note**: While STARDUST-MedSAM2 can be used for segmenting various anatomical structures including organs, it has been specifically trained and optimized for gross tumor volume (GTV) segmentation in radiotherapy contexts.

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/yourusername/STARDUST-MedSAM2.git
cd STARDUST-MedSAM2

# Install dependencies
pip install -e .

# Install PyTorch 2.3.1+
Follow the official guide: [PyTorch Installation](https://pytorch.org/get-started/locally/)

# Download pretrained models
[Download here](https://your-download-link.com)
```

## 🔄 Data Preparation

### 📂 DICOM to NPZ Conversion

STARDUST-MedSAM2 works with medical images in NPZ format. To convert your DICOM images:

```bash
# Using a CSV file (traditional method):
python create_npz_files.py --input_dir /path/to/dicom_folder --output_dir /path/to/npz_output --csv_file /path/to/cases.csv

# Recursive scan of directories without CSV:
python create_npz_files.py --input_dir /path/to/dicom_folder --output_dir /path/to/npz_output --recursive
```

The script supports two modes of operation:
1. **CSV-based processing**: Uses a CSV file with patient information to locate and process specific DICOM folders
2. **Recursive directory scanning**: Automatically finds and processes any directory containing CT and RS DICOM files (multiple CT*.dcm files and one RS*.dcm file)

This utility can process DICOM files from multiple sources, including those exported from TPS systems, and generates pseudonymized data for research purposes.

For training/fine-tuning, convert NPZ to NPY format:

```bash
python npz_to_npy.py -npz_dir ./data/npz_files -npy_dir ./data/npy_data -target_label 1
```

## 💻 Usage

### 🎮 Interactive Interface

Launch the interactive segmentation GUI:

```bash
python stardust_gui.py
```

<div align="center">
  <img src="assests/gui_screenshot.png" alt="STARDUST-MedSAM2 GUI" width="800"/>
</div>

## 🛠️ Model Fine-tuning

For specialized applications, fine-tune the pre-trained models:

```bash
python finetune_sam2_img.py -i ./data/npyFromDicom -task_name MedSAM2-Tiny-DICOM -work_dir ./work_dir -batch_size 8 -num_epochs 500 -pretrain_model_path ./checkpoints/sam2_hiera_tiny.pt -model_cfg sam2_hiera_t.yaml -one_label_per_epoch false -device cuda:0 -generate_validation_images false
```

## 📊 Evaluation

Compute segmentation metrics:

```bash
python ./eval_metrics/compute_metrics.py -s ./segs/2D/medsam2 -g ./data/npz_files -csv_dir ./results_inf_metric/2D/medsam2
```

## 🔮 Future Development

- 🔄 DICOM node for clinical workflow integration
- 📊 Enhanced metadata processing
- 🎯 Enhanced prompt mechanisms for clinical workflows

## 📚 Citation

If you use STARDUST-MedSAM2 in your research, please cite:

```
@article{STARDUST-MedSAM2,
  title={STARDUST-MedSAM2},
  author={Your Name},
  journal={Your Journal},
  year={2025}
}
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- This project builds upon [MedSAM2](https://github.com/bowang-lab/MedSAM/tree/MedSAM2) by Bo Wang's Lab
- SAM2 developed by Meta Platforms, Inc.
