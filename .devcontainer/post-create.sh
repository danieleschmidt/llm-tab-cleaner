#!/bin/bash
set -e

echo "🚀 Setting up LLM Tab Cleaner development environment..."

# Install Python dependencies in development mode
pip install -e .[dev] || pip install -e .

# Set up pre-commit hooks
pre-commit install --install-hooks

# Create .env file from template if it doesn't exist
if [ ! -f .env ]; then
    cp .env.example .env
    echo "📝 Created .env file from template - please update with your settings"
fi

# Set up git hooks
git config --global --add safe.directory /workspace

# Create necessary directories
mkdir -p logs
mkdir -p data/cache
mkdir -p data/temp

# Initialize database (if needed)
# python scripts/init_db.py

echo "✅ Development environment setup complete!"
echo ""
echo "📋 Next steps:"
echo "  1. Update .env file with your API keys and settings"
echo "  2. Run: make test   (to run tests)"
echo "  3. Run: make dev    (to start development server)"
echo "  4. Run: make lint   (to check code quality)"
echo ""
