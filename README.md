# AI Based Product Review Summarizer

This project utilizes the power of Large Language Models (LLMs) to generate concise, insightful summaries of product reviews. It leverages the Llama-2-7b-chat model from Meta to analyze and synthesize information from multiple product reviews.

## Features

- Utilizes the Llama-2-7b-chat model for high-quality text generation
- Processes reviews in batches for efficient summarization
- Generates summaries that include:
  - Brief product overview
  - Main product features
  - Semantic analysis of reviews (positive, negative, neutral)
  - Concluding statement
- Handles large datasets of product reviews

## Requirements

- Python 3.7+
- transformers
- torch
- langchain
- pandas
- einops
- accelerate
- bitsandbytes
- sentencepiece

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/product-review-summarizer.git
   cd product-review-summarizer
   ```

2. Install the required packages:
   ```
   pip install -q transformers einops accelerate langchain bitsandbytes sentencepiece accelerate langchain_community
   ```

3. Login to Hugging Face (required to access the Llama-2 model):
   ```
   huggingface-cli login
   ```

## Usage

1. Prepare your dataset:
   - Ensure your review data is in a CSV file
   - The CSV should have a column containing the review text

2. Update the script with your file path and column name:
   ```python
   df = pd.read_csv('path/to/your/file.csv')
   reviews = df['your_review_column_name'].tolist()
   ```

3. Run the script:
   ```
   python summarizer.py
   ```

4. The script will process reviews in batches and print summaries to the console.

## How it Works

1. The script uses the Llama-2-7b-chat model to generate summaries.
2. It processes reviews in batches to manage memory usage and improve efficiency.
3. For each batch, it generates a summary following these guidelines:
   - Introduction with a brief product overview
   - Main points summarizing key product features
   - Semantic analysis of the reviews
   - Conclusion with a final statement about the product

## Customization

You can adjust several parameters in the script:

- `batch_size`: Change the number of reviews processed in each batch
- `max_length`: Adjust the maximum length of generated summaries
- `temperature`: Modify the randomness of the output (lower for more deterministic results)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
