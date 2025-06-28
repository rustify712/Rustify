# Rustify: Towards Repository-Level C to Safer Rust via Workflow-Guided Multi-Agent Transpiler

**Rustify** is a tool for modernizing legacy C/C++ codebases by automatically translating them into safe and concurrent Rust code. By leveraging the power of Clang's Abstract Syntax Tree (AST) parsing and a sophisticated workflow driven by Large Language Models (LLMs), Rustify tackles the complexities of C/C++ to Rust migration, including syntax mapping, type safety conversion, and memory management challenges. This project aims to help developers enhance the safety, maintainability, and performance of their software.

This repository contains the source code and resources for our research paper.

## Features

- **Automated Code Translation**: Translates C/C++ code to idiomatic Rust.
- **AST-based Analysis**: Uses `libclang` to parse C/C++ source code into an AST for accurate and deep structural understanding.
- **LLM-Powered Workflow**: Employs a multi-agent system powered by LLMs to reason about and refactor the code, ensuring high-quality translation.
- **Safety and Concurrency**: Focuses on generating safe Rust code by handling pointers, memory allocation, and concurrency patterns.

## Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.9+
- Clang 14 (`libclang.so`)
- Rust toolchain (`rustc`, `cargo`)

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/rustify712/Rustify.git
    cd Rustify
    ```

2.  Install the required Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

**IMPORTANT**: Before running the tool, you must configure your settings in `core/config.py`.

You need to update the following critical configurations:

- `CLANG_LIB_FILE`: Path to your `libclang.so` file.
- `LLM_CONFIGS`: Configure your LLM provider details (e.g., OpenAI, Anthropic), including API keys, base URLs, and model names.
- `RAG_CONFIG`: If you are using Retrieval-Augmented Generation, configure your embedding model details.
- `DB_URL`: The database connection string.

Here is an example snippet from `core/config.py` that you need to edit:
```python
class Config:
    CLANG_LIB_FILE = "/usr/lib/llvm-14/lib/libclang.so"
    # ...

    LLM_CONFIGS = [
        {
            "provider": "openai",
            "model": "<your-model>",
            "base_url": "<your-provider-base-url>",
            "api_key": "<your-api-key>",
            # ...
        },
        # ...
    ]

    RAG_CONFIG = {
        "base_url": "<your-embedding-model-base-url>",
        "api_key": "<your-embedding-model-api-key>",
        "model": "<your-embedding-model>",
        # ...
    }
    # ...
```

## Usage

Once the configuration is complete, you can run Rustify on your C project:

```bash
python main.py -p /path/to/your/c-project
```

## License

This project is licensed under the MIT License.