# REPAIR-OF-THOUGHT: Automated Program Repair Framework

**ROT** is a sophisticated, UI-based Automated Program Repair (APR) tool that leverages the power of Large Language Models (LLMs) to generate repair suggestions and patches for bugs in software projects. Inspired by cutting-edge research, ROT is designed to address function-level bug repair, offering a streamlined approach that reduces the need for costly, statement-level fault localization.

## Overview

- **Focus**: Function-level Automated Program Repair (APR)
- **Framework**: UI-driven solution built with Streamlit for easy interaction
- **Objective**: Repair buggy code by generating solutions and patches for function-level bugs, primarily in the Defects4J dataset.

ROT is capable of generating potential fixes for single-function bugs through a dual-stage process of solution generation and patch creation. Unlike traditional APR techniques that focus on line-level or hunk-level repair, ROT takes a function-level approach, making it more efficient and practical in real-world scenarios. It aims to surpass existing APR methods by leveraging the extensive learning capabilities of LLMs.

## Key Features

- **Powerful LLM Integration**: Uses LLMs to generate natural language repair suggestions and patches, following a Chain of Thought (CoT) approach.
- **Function-Level Repair**: Goes beyond the conventional line-level or hunk-level techniques, focusing instead on entire function blocks to achieve more impactful repairs.
- **User Interface**: Built with Streamlit, the application offers an intuitive UI that simplifies the APR process for users.
- **Cost-Efficiency**: Achieves robust repair performance without requiring detailed statement-level fault localization, significantly reducing costs.

## 🔧 ROT Framework

The ROT workflow consists of the following stages:

1. **Solution Suggestion Generation**: The model takes a buggy function as input, utilizes auxiliary repair-relevant information (e.g., error messages, test reports, comments), and outputs natural language repair suggestions using the Chain of Thought (CoT) approach.
2. **Patch Generation**: The suggestions from the previous stage are used to auto-generate patches for the buggy function. The resulting patches are compiled and tested to validate the fixes.

The framework structure is designed to maximize LLM utility by enhancing both the generation of repair suggestions and the generation of the entire patched function.

## 📂 Dataset

### Defects4J 1.2 & 2.0

- **Defects4J**: A collection of 835 real-world bugs extracted from open-source Java projects. ROT focuses on function-level bugs, covering 522 single-function bugs in its initial phase.

### QuixBugs (Planned Integration)

- **QuixBugs**: Includes buggy and fixed versions of classic programming problems in Python and Java. The integration of this dataset is planned for future versions of ROT, expanding the scope of repair capabilities.

## 📊 Evaluation Metrics

- **Plausible Patch**: A patch that passes all relevant test cases, including both trigger and class-level tests.
- **Correct Patch**: A patch that is semantically equivalent to the developer's original fix.

The ROT evaluation reveals significant outperformance over traditional methods, particularly when tackling function-level bugs without relying on fine-grained statement-level localization.

## ⚙️ Environment Setup

### System Requirements

- **OS**: Linux with support for Docker (Optional: NVIDIA Docker for GPU acceleration).
- **Python**: Python 3.8 or later.
- **Dependencies**: See `requirements.txt` for complete dependency setup.

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/ROT.git
   cd ROT
   ```

2. **Install required dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:

   ```bash
   streamlit run app.py
   ```

### Optional: Docker Setup

For users wishing to containerize the environment, a Dockerfile is included:

```bash
docker build ./ --tag rot-apr
```

To run the container:

```bash
docker run -it --name rot-apr-container rot-apr
```

## 🚀 Quick Start

### Generating Solutions and Patches

1. Launch the Streamlit interface with `streamlit run app.py`.
2. Specify the bug ID (e.g., `Math-2`) and sample size (number of suggestions to generate).
3. Click on "Generate Solution and Patch" to start the process.
4. The application will generate solutions, extract relevant information, and create patches.
5. The generated solutions and patches can be viewed directly in the UI.

### Detailed Commands

To generate a solution for a bug using the command line, run the following:

```bash
python src/sf_gen_solution.py -d datasets/defects4j-sf.json -o outputs/sol/<bug_id>.json -s 2 -bug <bug_id>
```

To generate patches based on extracted solutions:

```bash
python src/sf_gen_patch.py -d datasets/defects4j-sf.json -s outputs/sol/<bug_id>_extracted.json -o outputs/patches/<bug_id>_patch.json -bug <bug_id>
```

## 🏆 Repair Performance

ROT achieves remarkable results in fixing function-level bugs:
- Correctly fixes 300 out of 522 single-function bugs in the Defects4J dataset.
- The average cost per fixed bug is $0.029, making it highly efficient compared to traditional methods.
- Successfully generates solutions that outperform the state-of-the-art APR techniques such as ChatRepair and Repilot, without the need for detailed fault localization.

## ⚡️ Planned Features

- **Multi-Function Bug Repair**: Extend capabilities to handle bugs spanning multiple functions.
- **Patch Validation Module**: A robust validation pipeline for verifying the semantic correctness of generated patches.
- **Support for More Languages**: Integration of datasets like QuixBugs to expand language support beyond Java.

## 🤖 Technology Stack

- **Frontend**: Streamlit for user interaction.
- **Backend**: Python, Subprocess for executing APR workflows.
- **APR Engine**: LLM-based solution generation and patch creation.
- **Dataset**: Defects4J and QuixBugs.

## 🔗 References
- "How Far Can We Go with Practical Function-Level Program Repair?" - Xiang et al.
- SRepair: "Powerful LLM-based Program Repairer with $0.029/Fixed Bug"

## 👥 Contributors

- **Satwik Pandey** - [GitHub](https://github.com/satwik-pandey)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---


