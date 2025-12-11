#!/bin/bash

# System Validation Script
# Checks if the RAG system is properly configured

set -e

echo "=================================="
echo "RAG System - Validation Script"
echo "=================================="
echo ""

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Check file structure
echo "Checking project structure..."

required_files=(
    "docker-compose.yml"
    ".env.example"
    ".gitignore"
    "README.md"
    "backend/Dockerfile"
    "backend/requirements.txt"
    "backend/app/main.py"
    "backend/config/settings.py"
    "backend/models/database.py"
    "backend/models/schemas.py"
    "backend/services/embeddings.py"
    "backend/services/reranker.py"
    "backend/services/document_processor.py"
    "backend/services/rag_service.py"
    "backend/services/file_handler.py"
    "frontend/Dockerfile"
    "frontend/index.html"
    "frontend/style.css"
    "frontend/script.js"
    "frontend/nginx.conf"
)

missing_files=0
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        print_success "$file exists"
    else
        print_error "$file is missing"
        missing_files=$((missing_files + 1))
    fi
done

echo ""

if [ $missing_files -gt 0 ]; then
    print_error "$missing_files required files are missing"
    exit 1
else
    print_success "All required files are present"
fi

echo ""

# Check Python syntax
echo "Checking Python syntax..."
python_files=$(find backend -name "*.py" 2>/dev/null)
python_errors=0

for file in $python_files; do
    if python3 -m py_compile "$file" 2>/dev/null; then
        print_success "$file syntax OK"
    else
        print_error "$file has syntax errors"
        python_errors=$((python_errors + 1))
    fi
done

echo ""

if [ $python_errors -gt 0 ]; then
    print_error "$python_errors Python files have syntax errors"
    exit 1
else
    print_success "All Python files have valid syntax"
fi

echo ""

# Check Docker Compose configuration
echo "Checking Docker Compose configuration..."

if command -v docker &> /dev/null; then
    if docker compose config &> /dev/null; then
        print_success "docker-compose.yml is valid"
    else
        print_error "docker-compose.yml has errors"
        exit 1
    fi
else
    print_warning "Docker not found, skipping compose validation"
fi

echo ""

# Check .env file
echo "Checking environment configuration..."

if [ -f .env ]; then
    print_success ".env file exists"
    
    if grep -q "your-api-key-here" .env; then
        print_warning "LLM_API_KEY is still set to placeholder. Update it before running queries."
    else
        print_success "LLM_API_KEY appears to be configured"
    fi
else
    print_warning ".env file not found. Copy .env.example to .env and configure it."
fi

echo ""

# Check if Docker is running
echo "Checking Docker daemon..."

if docker info &> /dev/null; then
    print_success "Docker is running"
else
    print_warning "Docker is not running. Start Docker to use the system."
fi

echo ""

# Check if services are running
echo "Checking if services are running..."

if docker compose ps 2>/dev/null | grep -q "Up"; then
    print_success "Some services are running"
    
    services=("postgres" "qdrant" "backend" "frontend")
    for service in "${services[@]}"; do
        if docker compose ps 2>/dev/null | grep "$service" | grep -q "Up"; then
            print_success "$service is running"
        else
            print_warning "$service is not running"
        fi
    done
else
    print_warning "No services are running. Run 'docker compose up' to start."
fi

echo ""

# Check if ports are available
echo "Checking port availability..."

ports=("3000:Frontend" "8000:Backend" "5432:PostgreSQL" "6333:Qdrant")
for port_info in "${ports[@]}"; do
    port="${port_info%%:*}"
    service="${port_info##*:}"
    
    if command -v lsof &> /dev/null; then
        if lsof -Pi :$port -sTCP:LISTEN -t &>/dev/null; then
            if docker compose ps 2>/dev/null | grep -q "0.0.0.0:$port"; then
                print_success "Port $port ($service) is in use by Docker"
            else
                print_warning "Port $port ($service) is in use by another process"
            fi
        else
            print_success "Port $port ($service) is available"
        fi
    else
        print_warning "lsof not available, skipping port check for $port"
    fi
done

echo ""

# Summary
echo "=================================="
echo "Validation Summary"
echo "=================================="
echo ""

if [ $missing_files -eq 0 ] && [ $python_errors -eq 0 ]; then
    print_success "System structure is valid"
    echo ""
    echo "Next steps:"
    echo "1. Ensure .env is configured with your LLM API key"
    echo "2. Run: ./start.sh (or 'docker compose up --build')"
    echo "3. Access: http://localhost:3000"
    echo ""
else
    print_error "System validation failed"
    echo ""
    echo "Please fix the errors above before running the system."
    exit 1
fi
