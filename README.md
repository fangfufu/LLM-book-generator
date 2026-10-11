# LLM Book Generator 📚🤖

LLM Book Generator is a powerful Python-based command-line tool that automates
the creation of entire books, from initial concept to a fully formatted DOCX
manuscript. It leverages Large Language Models (LLMs) via llama.cpp to generate
diverse content, including titles, chapter outlines, detailed section text,
front and back matter, and even marketing materials.

## Features

- **llama.cpp Support:** Seamlessly integrates with llama.cpp server
  (`llama-server`).
- **Comprehensive Content Generation:** Creates everything from titles and
  chapter outlines to front and back matter.
- **LaTeX Math Rendering:** Converts LaTeX math expressions into high-quality
  images.
- **Advanced DOCX Styling:** Produces professionally formatted DOCX files with
  customizable styles.
- **Intelligent Caching:** Caches LLM responses to speed up subsequent runs.
- **Flexible Configuration:** Uses a `config.yaml` file for easy customization.

## Getting Started

### Prerequisites

- Python 3.12+
- A LaTeX distribution (e.g., [MiKTeX](https://miktex.org/),
  [MacTeX](https://www.tug.org/mactex/),
  [TeX Live](https://www.tug.org/texlive/))
- A running [llama.cpp server](https://github.com/ggerganov/llama.cpp) instance
  (`llama-server`).

### Installation

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install dependencies:**

   ```bash
   pip install .
   ```

3. **Set up environment (Optional):** If using Hugging Face tokenizers requiring
   authentication, create a `.env` file in the project root:

   ```
   HF_TOKEN="YOUR_HUGGING_FACE_TOKEN"
   ```

## How to Run

1. **Configure your book:** Edit the `config.yaml` file to define your book's
   topic, author, style, and other parameters. For a detailed explanation of all
   the available options, see [CONFIG.md](CONFIG.md).

2. **Run the script:**

   ```bash
   python main.py
   ```

   The generated book and marketing materials will be saved in the directory
   specified in `config.yaml` (default is `"output"`).

## Troubleshooting

- **LaTeX Math Rendering Issues:** Ensure your LaTeX distribution is installed
  correctly and accessible from your system's PATH.
- **API Errors:** Ensure that your llama.cpp server is running and accessible
  (default: `http://localhost:8080`).
- **File Permissions:** Make sure the script has write access to the output and
  cache directories.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an
issue.

## License

This project is licensed under GPLv3. See the [LICENSE](LICENSE) file for
details.
