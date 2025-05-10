# AIVA - Artificial Intelligence Voice Accountant API

AIVA (Artificial Intelligence Voice Accountant) is a powerful API that processes natural language financial requests and converts them into structured data and SQL statements. It helps users track expenses, analyze spending patterns, and extract data from receipts using AI.

## Features

- **Natural Language Parsing**: Convert casual text like "I spent $50 on groceries yesterday" into structured data
- **Receipt OCR Processing**: Extract expense information from receipt text
- **Financial Command Processing**: Generate SQL queries for financial analysis based on natural language commands
- **SQL Generation**: Create SQL statements for a client-side SQLite database
- **SQL Explanation**: Generate plain-language explanations of SQL queries

## Tech Stack

- Python 3.10+
- FastAPI
- LangChain/LangGraph
- OpenAI API
- Pydantic
- SQLAlchemy (for database models)
- Loguru (for structured logging)

## Installation

### Prerequisites

- Python 3.10 or higher
- An OpenAI API key

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/AnalyticAce/AIVA.git
   cd AIVA
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -e .
   ```

4. Create a `.env` file based on `.env.example`:
   ```bash
   cp .env.example .env
   ```

5. Edit the `.env` file and add your OpenAI API key and other settings.

## Usage

### Starting the API Server

Run the following command to start the API server:

```bash
python main.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`, and interactive documentation can be accessed at `http://localhost:8000/api/docs`.

### API Endpoints

#### 1. Parse Natural Language Text

```http
POST /api/v1/agent/parse
```

Example request:
```json
{
  "text": "I spent $42.50 on groceries at Whole Foods yesterday",
  "context": {
    "user_timezone": "America/New_York",
    "currency": "USD"
  }
}
```

Example response:
```json
{
  "intent": "add_expense",
  "entities": {
    "amount": 42.50,
    "category": "groceries",
    "date": "2025-05-07",
    "store": "Whole Foods"
  },
  "sql": "INSERT INTO expenses (amount, category, date, store) VALUES (42.50, 'groceries', '2025-05-07', 'Whole Foods');",
  "summary": "Added a 42.5 expense to groceries at Whole Foods for 2025-05-07."
}
```

#### 2. Parse Receipt Text

```http
POST /api/v1/agent/receipt
```

Example request:
```json
{
  "receipt_text": "WHOLE FOODS MARKET\n123 Main St\nNew York, NY 10001\n\nDate: 05/07/2025\n\nOrganic Bananas    $3.99\nAlmond Milk        $4.50\nAvocado            $2.50\nWhole Grain Bread  $5.99\n\nSubtotal:         $16.98\nTax:               $1.52\nTotal:            $18.50\n\nThank you for shopping with us!",
  "context": {
    "user_timezone": "America/New_York",
    "currency": "USD"
  }
}
```

Example response:
```json
{
  "intent": "add_expense",
  "entities": {
    "amount": 18.50,
    "category": "groceries",
    "date": "2025-05-07",
    "store": "WHOLE FOODS MARKET"
  },
  "sql": "INSERT INTO expenses (amount, category, date, store) VALUES (18.50, 'groceries', '2025-05-07', 'WHOLE FOODS MARKET');",
  "summary": "Added a 18.5 expense to groceries at WHOLE FOODS MARKET for 2025-05-07."
}
```

#### 3. Process Financial Command

```http
POST /api/v1/agent/command
```

Example request:
```json
{
  "text": "Show me how much I spent on groceries last month",
  "context": {
    "user_timezone": "America/New_York",
    "currency": "USD"
  }
}
```

Example response:
```json
{
  "intent": "query_category_expenses",
  "sql": "SELECT * FROM expenses WHERE category = 'groceries' AND date >= DATE('now', '-30 day') ORDER BY date DESC;",
  "summary": "This query retrieves all columns from expenses in the groceries category for the last 30 days ordered by date in descending order."
}
```

### Authentication

The API uses API key authentication. To authenticate requests, include the API key in the `X-API-KEY` header (configurable in `.env`).

## Project Structure

```
aiva/
├── api/                 # API endpoints
│   └── v1/              # Version 1 endpoints
│       └── endpoints/   # Endpoint implementation
├── core/                # Core configuration
├── db/                  # Database models and utilities
├── schemas/             # Pydantic models for request/response
├── services/            # Business logic
│   ├── agents/          # LLM agents implementation
│   └── tools/           # Agent tools
└── utils/               # Utility functions
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| OPENAI_API_KEY | OpenAI API key | - |
| OPENAI_MODEL | OpenAI model to use | gpt-4-turbo |
| API_HOST | Host to run the API on | 0.0.0.0 |
| API_PORT | Port to run the API on | 8000 |
| DEBUG | Enable debug mode | False |
| API_KEY_HEADER | Name of the API key header | X-API-KEY |
| API_KEY | API key for authentication | - |
| LOG_LEVEL | Logging level | INFO |

## Development

### Testing

To run tests:

```bash
pytest
```

### Code Quality

To check code quality with ruff:

```bash
ruff check .
```

To format code:

```bash
black .
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgements

- OpenAI for providing the language models
- The FastAPI and LangChain communities for their excellent tools