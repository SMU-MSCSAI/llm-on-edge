# LLM-on-Edge

This repository contains the work for researching and understanding the deployment of large language models (LLMs) on edge devices.

## Getting Started

This guide will help you set up your environment, select and quantize a model, and implement the initial phase of your project.

### Prerequisites

Ensure you have the following installed on your system:
- Anaconda or Miniconda
- Python 3.9
- Git

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/llm-on-edge.git
   cd llm-on-edge

2. **Create and activate a Conda environment**
# Create a new conda environment named 'llm_quant'
conda create --name llm_quant python=3.9

# Activate the environment
conda activate llm_quant

3. **Install the required dependencies**
# Install the required libraries from requirements.txt
pip install -r requirements.txt

## Tips

Tip1: To update your library run the following command after you've updated their version in the requirement.txt
```bash
pip install -r requirements.txt --upgrade
```


### Error Handling
1. Add your env into the jupyter kernel, run the following command
```python -m ipykernel install --user --name=llm_quant --display-name "Python (llm_quant)"```