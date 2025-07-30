#!/bin/bash

# Post-create script for LLM Tab Cleaner development container
# This script runs after the container is created to set up the development environment

set -e

echo "🚀 Setting up LLM Tab Cleaner development environment..."

# Ensure we're in the correct directory
cd /app

# Create history directory for command persistence  
mkdir -p ~/.history

# Set up git configuration if not already set
if [ -z "$(git config --global user.name)" ]; then
    echo "⚙️  Setting up Git configuration..."
    git config --global user.name "Dev Container User"
    git config --global user.email "dev@example.com"
    git config --global init.defaultBranch main
    git config --global pull.rebase false
fi

# Install development dependencies
echo "📦 Installing development dependencies..."
pip install --no-cache-dir -e ".[dev,test]"

# Install pre-commit hooks
echo "🪝 Installing pre-commit hooks..."
pre-commit install

# Verify installation
echo "✅ Verifying installation..."
python -c "from llm_tab_cleaner import TableCleaner; print('✅ LLM Tab Cleaner imported successfully')"

# Run quick tests to ensure everything is working
echo "🧪 Running quick test suite..."
pytest tests/ -x -q --tb=short

# Set up shell configuration
echo "🐚 Configuring shell..."

# Create .zshrc if it doesn't exist
if [ ! -f ~/.zshrc ]; then
    cp /root/.oh-my-zsh/templates/zshrc.zsh-template ~/.zshrc
fi

# Add custom aliases and functions to .zshrc
cat >> ~/.zshrc << 'EOF'

# LLM Tab Cleaner development aliases
alias ll='ls -la'
alias lt='ls -lat'
alias pytest-cov='pytest --cov=src --cov-report=term-missing --cov-report=html'
alias lint='make lint'
alias format='make format'
alias test='make test'
alias test-fast='make test-fast'
alias type-check='make type-check'
alias clean='make clean'
alias docs='make docs-serve'

# Python development helpers
alias pipi='pip install'
alias pipir='pip install -r'
alias activate='source venv/bin/activate'

# Git aliases
alias gst='git status'
alias gco='git checkout'
alias gcb='git checkout -b'
alias gaa='git add .'
alias gcm='git commit -m'
alias gps='git push'
alias gpl='git pull'
alias glog='git log --oneline --graph --decorate --all'

# Development shortcuts
alias run-example='python examples/basic_usage.py'
alias start-jupyter='jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root'

# Quick development commands
function test-file() {
    pytest "tests/test_$1.py" -v
}

function lint-file() {
    ruff check "src/llm_tab_cleaner/$1.py"
    black --check "src/llm_tab_cleaner/$1.py"
}

function format-file() {
    black "src/llm_tab_cleaner/$1.py"
    ruff check --fix "src/llm_tab_cleaner/$1.py"
}

# Environment info
echo "🔧 LLM Tab Cleaner Development Environment"
echo "Python: $(python --version)"
echo "Project: $(pwd)"
echo "Git branch: $(git branch --show-current 2>/dev/null || echo 'Not a git repository')"
echo ""
echo "Available commands:"
echo "  make test      - Run test suite"
echo "  make lint      - Run linting"
echo "  make format    - Format code"
echo "  make docs      - Build and serve documentation"
echo "  pytest-cov    - Run tests with coverage"
echo ""
EOF

# Create useful development directories
mkdir -p examples notebooks temp logs data

# Create example script if it doesn't exist
if [ ! -f examples/basic_usage.py ]; then
    cat > examples/basic_usage.py << 'EOF'
"""Basic usage example for LLM Tab Cleaner."""

import pandas as pd
from llm_tab_cleaner import TableCleaner

def main():
    """Run basic cleaning example."""
    # Create sample messy data
    df = pd.DataFrame({
        'name': ['Alice Johnson', 'Bob Smith', 'Charlie Brown'],
        'email': ['alice@email.com', 'bob.smith@company', 'charlie@test.co'],
        'age': ['25', 'thirty', '35'],
        'city': ['New York', 'SF', 'Los Angeles']
    })
    
    print("Original data:")
    print(df)
    print()
    
    # Initialize cleaner
    cleaner = TableCleaner(
        llm_provider="anthropic",  # Change as needed
        confidence_threshold=0.85
    )
    
    # Clean the data
    cleaned_df, report = cleaner.clean(df)
    
    print("Cleaned data:")
    print(cleaned_df)
    print()
    
    print("Cleaning report:")
    print(f"Total fixes: {report.total_fixes}")
    print(f"Quality score: {report.quality_score:.2%}")

if __name__ == "__main__":
    main()
EOF
fi

# Set up Jupyter configuration
mkdir -p ~/.jupyter
cat > ~/.jupyter/jupyter_lab_config.py << 'EOF'
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8888
c.ServerApp.open_browser = False
c.ServerApp.allow_root = True
c.ServerApp.token = ''
c.ServerApp.password = ''
EOF

# Create sample notebook
if [ ! -f notebooks/llm_cleaning_example.ipynb ]; then
    cat > notebooks/llm_cleaning_example.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# LLM Tab Cleaner Example\n",
    "\n",
    "This notebook demonstrates basic usage of the LLM Tab Cleaner library."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from llm_tab_cleaner import TableCleaner\n",
    "\n",
    "# Create sample data\n",
    "df = pd.DataFrame({\n",
    "    'name': ['Alice', 'Bob', 'Charlie'],\n",
    "    'age': [25, 30, 35]\n",
    "})\n",
    "\n",
    "print(\"Sample data created:\")\n",
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Initialize and run cleaner\n",
    "cleaner = TableCleaner()\n",
    "cleaned_df, report = cleaner.clean(df)\n",
    "\n",
    "print(f\"Cleaning completed with {report.total_fixes} fixes\")\n",
    "cleaned_df"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF
fi

echo "✅ Development environment setup complete!"
echo ""
echo "🎉 Ready to develop LLM Tab Cleaner!"
echo "   Run 'make test' to verify everything is working"
echo "   Run 'make docs' to start the documentation server"
echo "   Run 'start-jupyter' to launch Jupyter Lab"
echo "   Run 'run-example' to test basic functionality"
echo ""
EOF