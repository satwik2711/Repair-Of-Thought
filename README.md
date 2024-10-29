# REPAIR-OF-THOUGHT: Automated Program Repair Framework

**Repair of Thought** is a sophisticated Automated Program Repair (APR) tool that leverages the power of open sourced Large Language Models (LLMs) to generate repair suggestions and patches for most common bugs in software projects. Inspired by cutting-edge research, ROT is designed to address function-level bug repair, offering a streamlined approach that reduces the need for costly, statement-level fault localization. The core of ROT revolves around reasoning chains inspired from OpenAI’s current segment leader reasoning model O1, enabling a thorough understanding of bugs before generating effective solutions.

## Overview

- **Focus**: Function-level Automated Program Repair (APR) using open source models [LLAMA 3.1 70B, CODE-LLAMA 34B INSTRUCT]
- **Objective**: Repair buggy code by generating solutions and patches for function-level bugs, primarily in the Defects4J dataset.
- **Reasoning-Based Approach**: Utilizes reasoning chains through OpenAI’s GPT-4o or Llama models to thoroughly analyze bugs before generating fixes.

ROT is capable of generating potential fixes for single-function bugs through a dual-stage process of reasoning-based solution generation and patch creation. Unlike traditional APR techniques that focus on line-level or hunk-level repair, ROT takes a function-level approach, making it more efficient and practical in real world scenarios. It aims to surpass existing APR methods by leveraging the extensive learning capabilities of LLMs.

## Key Features

- **Powerful LLM Integration**: Uses LLMs to generate natural language repair suggestions and patches, following a Chain of Thought (CoT) approach.
- **Function-Level Repair**: Goes beyond the conventional line-level or hunk-level techniques, focusing instead on entire function blocks to achieve more impactful repairs.
- **User Interface**: Built with Streamlit, the application offers an intuitive UI that simplifies the APR process for users.
- **Cost-Efficiency**: Achieves robust repair performance without requiring detailed statement-level fault localization, significantly reducing costs.
- **Open Source and Cost-Free Options**: Provides both open-source and GPT-based reasoning chains, making it possible to choose between cost-effective Groq models (free) or GPT-4o (paid) for enhanced flexibility.

## 🔧 ROT Framework

The ROT workflow consists of the following stages:

1. **Reasoning Chain Generation**: Leveraging OpenAI’s current segment leader model (O1) or Groq (G1) for reasoning, this step lays down chain-of-thought processes to thoroughly understand the problem statement. The reasoning can be done using either GPT-4o (paid) or Groq (free), allowing users to select based on their requirements.
2. **Solution Suggestion Generation**: The reasoning is combined to form a final natural language solution. This solution is then passed to the Code-specific language model – Code Llama 34b Instruct[through TogetherAI] – to generate potential fixes.
3. **Patch Generation**: The suggestions from the previous stage are used to auto-generate patches for the buggy function. The resulting patches are compiled and tested to validate the fixes.

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
   git clone https://github.com/satwik2711/Repair-Of-Thought.git
   cd Repair-Of-Thought
   ```

2. **Install required dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Export your API keys**
   ```bash
   export TOGETHER_API_KEY=your_api_key
   export GROQ_API_KEY=your_api_key
   export OPENAI_API_KEY=your_api_key #only if using gpt4o reasoning
   ```

4. **Run the application**:

   ```bash
   streamlit run app-4o1.py  # For GPT-4o reasoning
   streamlit run app-g1.py   # For Groq reasoning (free version)
   ```

## 🚀 Quick Start

### Generating Solutions and Patches

1. Launch the Streamlit interface with either `streamlit run app-4o1.py` or `streamlit run app-g1.py`.
2. Specify the bug ID (e.g., `Math-2`) and sample size (number of suggestions to generate).
3. Click on "Generate Solution and Patch" to start the process.
4. The application will generate solutions, extract relevant information, and create patches.
5. The generated solutions and patches can be viewed directly in the UI.

### Detailed Commands

To generate a solution for a bug using the command line, run the following:

```bash
python src/sf_gen_solution_reasoned_gpt.py -d datasets/defects4j-sf.json -o outputs-4o1/sol/<bug_id>.json -s 2 -bug <bug_id>
```
or
```bash
python src/sf_gen_solution_reasoned.py -d datasets/defects4j-sf.json -o outputs-g1/sol/<bug_id>.json -s 2 -bug <bug_id>
```

To generate patches based on extracted solutions:

```bash
python src/sf_gen_patch.py -d datasets/defects4j-sf.json -s outputs-g1/sol/<bug_id>_extracted.json -o outputs-4o1/patches/<bug_id>_patch.json -bug <bug_id>
```

## 🏆 Repair Performance

ROT achieves remarkable results in fixing function-level bugs:
- Aims to correctly fix over 300 out of 522 single-function bugs in the Defects4J dataset.
- The average cost per fixed bug would be a little over $0.029 using GPT-4o, whereas using Groq-based models makes the cost effectively **$0 per bug**.
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
- **Reasoning Models**: OpenAI GPT-4o or Llama-3.1-70b for generating reasoning steps.
- **Code Model**: Code Llama 34b Instruct for code patch generation.

## 🔗 References
- "How Far Can We Go with Practical Function-Level Program Repair?" - Xiang et al.
- SRepair: "Powerful LLM-based Program Repairer with $0.029/Fixed Bug"

## 👥 Contributors

- **Satwik Pandey** - [GitHub](https://github.com/satwik-pandey)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


