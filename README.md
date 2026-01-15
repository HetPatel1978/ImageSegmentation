# Image Segmentation

A comprehensive project exploring multiple approaches to image segmentation and document processing, with integration of text classification capabilities.

## 📋 Overview

This repository implements various image segmentation techniques for analyzing and processing document images. The project includes multiple methods for segmenting images, extracting text, and classifying content from PDF documents.

## 🎯 Features

- **Multiple Segmentation Methods**: Four different approaches to image segmentation (Method 1-4)
- **Document Processing**: PDF to image conversion and analysis
- **Text Classification**: Integrated text classification pipeline
- **Detectron2 Integration**: Implementation using Mask R-CNN for instance segmentation
- **Preprocessing Pipeline**: Text preprocessing and annotation tools

## 📁 Project Structure

```
ImageSegmentation/
├── Method-1/              # First segmentation approach
├── Method-2/              # Second segmentation approach
├── Method-3/              # Third segmentation approach
├── Method-4/              # Fourth segmentation approach
├── Text_classification/   # Text classification module
├── Updated/               # Updated implementations and improvements
├── Results/               # Output results and visualizations
├── requirenment.txt       # Project dependencies
└── *.png, *.pdf, *.txt   # Sample files and results
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- pip package manager
- CUDA-capable GPU (recommended for deep learning methods)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/HetPatel1978/ImageSegmentation.git
cd ImageSegmentation
```

2. Install required dependencies:
```bash
pip install -r requirenment.txt
```

### Dependencies

The project uses several key libraries:
- OpenCV for image processing
- PyTorch for deep learning models
- Detectron2 for instance segmentation
- Tesseract OCR for text extraction
- scikit-learn for classification tasks

## 💻 Usage

### Basic Image Segmentation

Navigate to the desired method directory and run the corresponding notebook or script:

```bash
cd Method-1
jupyter notebook
```

### Processing PDF Documents

The project includes functionality to process PDF documents, extract text, and perform segmentation on individual pages.

### Text Classification

The text classification module can be used to categorize extracted text:

```bash
cd Text_classification
python classify.py
```

## 📊 Methods Overview

### Method 1
Traditional computer vision techniques using edge detection and contour analysis.

### Method 2
Threshold-based segmentation with morphological operations.

### Method 3
Region-based segmentation using clustering algorithms.

### Method 4
Deep learning approach using Detectron2 with Mask R-CNN architecture.

## 🎨 Results

Sample results and visualizations can be found in the `Results/` directory. The repository includes:
- Annotated page images
- Segmentation masks
- Processed text outputs
- Comparative analysis figures

## 📝 Examples

Example outputs include:
- `page_2_annotated.png`: Annotated document page with detected regions
- `Detectron2_usingRCNN.png`: Mask R-CNN segmentation results
- `preprocessed_text.txt`: Extracted and preprocessed text

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is available for educational and research purposes.

## 👤 Author

**Het Patel**
- GitHub: [@HetPatel1978](https://github.com/HetPatel1978)

## 🙏 Acknowledgments

- Detectron2 by Facebook AI Research
- OpenCV community
- PyTorch team

## 📧 Contact

For questions or feedback, please open an issue in the repository or contact the author directly through GitHub.

---

**Note**: This is an active research project. Methods and implementations are subject to updates and improvements.
