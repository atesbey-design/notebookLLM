# Podcast Generator

This project is a Python application that automates the process of extracting text from a PDF file, cleaning the text, and then converting it into an audio podcast. The project utilizes various libraries for text processing and speech synthesis.

## Project Structure

```
podcast-generator/
│
├── TTS.py                     # Main file for converting podcast text to audio
├── pdf-processing.py          # Extracts and processes text from PDF files
├── rewrite-podcast.py         # Rewrites the podcast transcript
├── clean-pdf.py               # Cleans the PDF file and extracts text
├── resources/                 # Project resource files
│   ├── podcast_ready_data.txt # Text prepared for the podcast
│   ├── ai-short.pdf           # PDF file to be processed
│   ├── extracted_text.txt     # Extracted text file
│   ├── data.pkl               # Extracted text and metadata information
│   └── _podcast.mp3           # Generated podcast file
└── README.md                  # Information about the project
```

## Requirements

The following Python libraries are required for the project to run:

- `torch`
- `transformers`
- `PyPDF2`
- `pydub`
- `tqdm`
- `accelerate`
- `numpy`
- `scipy`

You can install the required libraries using the following command:

```
pip install torch transformers PyPDF2 pydub tqdm accelerate numpy scipy
```

## Usage

1. **Processing the PDF File**: Run the `pdf-processing.py` file to extract text from the PDF file. The extracted text will be saved in the `extracted_text.txt` file.

   ```bash
   python pdf-processing.py
   ```

2. **Cleaning the Text**: Run the `clean-pdf.py` file to clean the extracted text and save it in the `data.pkl` file.

   ```bash
   python clean-pdf.py
   ```

3. **Converting Podcast Text to Audio**: Run the `TTS.py` file to convert the cleaned text into an audio podcast. The generated podcast file will be saved as `resources/_podcast.mp3`.

   ```bash
   python TTS.py
   ```

4. **Rewriting the Podcast Transcript**: Run the `rewrite-podcast.py` file to rewrite the podcast transcript. The rewritten text will be saved in the `podcast_ready_data.pkl` file.

   ```bash
   python rewrite-podcast.py
   ```

## Contributing

If you would like to contribute to this project, please feel free to create a pull request or share your suggestions.

## License

This project is licensed under the MIT License.