# Automata Solver

A web application for solving, analyzing, and visualizing automata problems with AI-powered explanations.

## Features

- Convert regular expressions to DFAs using Thompson's construction
- Minimize automata using Hopcroft's algorithm
- AI-powered explanations of automata concepts
- Batch processing capabilities
- Interactive automata visualization
- Real-time analysis and validation
- API documentation with Swagger/OpenAPI

## Architecture

- FastAPI backend with async support
- Redis for caching and rate limiting
- Prometheus and Grafana for monitoring
- Docker containerization
- GitHub Actions for CI/CD

## Getting Started

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- Redis

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/automata-solver.git
   cd automata-solver
   ```

2. Run the setup script:
   ```bash
   python scripts/setup.py
   ```

3. Configure environment variables:
   - Copy `.env.template` to `.env` in the backend directory
   - Update the values with your configuration

4. Start the development environment:
   ```bash
   docker-compose up -d
   ```

### Development

- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

### Running Tests

```bash
cd backend
pytest
```

## Project Structure

```
automata-solver/
├── backend/
│   ├── automata_solver.py    # Core automata logic
│   ├── ai_explainer.py       # AI integration
│   ├── server.py            # FastAPI application
│   ├── metrics.py           # Prometheus metrics
│   ├── security.py          # Authentication/Authorization
│   └── tests/               # Test suite
├── docker-compose.yml       # Development environment
├── prometheus.yml          # Prometheus configuration
└── scripts/               # Development scripts
```

## API Documentation

### Main Endpoints

- `POST /api/convert`: Convert regex to DFA
- `POST /api/minimize`: Minimize automaton
- `POST /api/analyze`: Analyze automaton properties
- `POST /api/bulk/convert`: Batch conversion
- `GET /metrics`: Prometheus metrics
- `GET /health`: Health check

## Monitoring

- Prometheus metrics include:
  - Request counts and latencies
  - Cache hit/miss rates
  - Resource utilization
  - Error rates
  - Business metrics (operations per automaton type)

- Grafana dashboards available for:
  - System monitoring
  - Application performance
  - Error tracking
  - User activity

## Security

- JWT-based authentication
- Rate limiting
- Input validation
- Secure configuration management
- CORS protection

## Contributing

1. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

2. Follow the code style guidelines
3. Write tests for new features
4. Submit pull requests with comprehensive descriptions

## License

This project is licensed under the MIT License - see the LICENSE file for details.