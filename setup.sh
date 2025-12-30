#!/bin/bash
set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

print_success() { echo -e "${GREEN}✓${NC} $1"; }
print_error() { echo -e "${RED}✗${NC} $1"; exit 1; }
print_info() { echo -e "${BLUE}→${NC} $1"; }
echo "========================================="
echo "  RAG System - Setup"
echo "========================================="


print_info "Checking Docker..."
docker info > /dev/null 2>&1 || print_error "Docker not running - please start Docker first"
print_success "Docker is running"

print_info "Checking configuration..."
[ ! -f .env ] && print_error ".env file missing - copy .env.example and configure OPENAI_API_KEY"
print_success "Configuration found"

mkdir -p backend/data backend/models

print_info "Starting RAG System..."
docker-compose up -d --build

echo ""
echo " Access: http://localhost:3000"
echo "  Backend:   http://localhost:8000"
docker compose logs -f backend
# TODO
# echo "  GPU:       watch -n 1 rocm-smi"
echo "========================================="