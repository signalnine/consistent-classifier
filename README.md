# Classify Texts

A Go application for text classification.

## Getting Started

### Prerequisites

- Go 1.19 or higher
- Git

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd classify-texts
   ```

2. Install dependencies:
   ```bash
   go mod tidy
   ```

### Running the Application

```bash
go run main.go
```

### Building the Application

```bash
go build -o bin/classify-texts main.go
```

## Project Structure

```
classify-texts/
├── main.go           # Main application entry point
├── go.mod           # Go module file
├── .gitignore       # Git ignore rules
└── README.md        # Project documentation
```

## Development

### Running Tests

```bash
go test ./...
```

### Code Formatting

```bash
go fmt ./...
```

### Code Linting

```bash
go vet ./...
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

