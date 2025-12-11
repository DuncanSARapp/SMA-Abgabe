#!/bin/bash

# Quick Start Script for RAG System
# This script helps you get the system up and running quickly

set -e

echo "=================================="
echo "RAG System - Quick Start"
echo "=================================="
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Creating from .env.example..."
    cp .env.example .env
    echo ""
    echo "⚠️  IMPORTANT: Please edit .env and add your LLM API key:"
    echo "   nano .env"
    echo ""
    echo "Press Enter after you've configured your API key..."
    read -r
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

echo "✓ Docker is running"
echo ""

# Check if API key is set
if grep -q "your-api-key-here" .env; then
    echo "⚠️  WARNING: LLM_API_KEY is still set to placeholder value"
    echo "   The system will start but queries will fail without a valid API key."
    echo ""
    echo "Continue anyway? (y/n)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "Please edit .env and add your API key, then run this script again."
        exit 0
    fi
fi

echo "🚀 Building and starting services..."
echo "   This may take a few minutes on first run..."
echo ""

# Build and start services
docker compose up --build -d

echo ""
echo "⏳ Waiting for services to be ready..."
sleep 10

# Check if services are running
if docker compose ps | grep -q "Up"; then
    echo ""
    echo "=================================="
    echo "✅ System is running!"
    echo "=================================="
    echo ""
    echo "Access points:"
    echo "  • Frontend:  http://localhost:3000"
    echo "  • Backend:   http://localhost:8000"
    echo "  • API Docs:  http://localhost:8000/docs"
    echo ""
    echo "To view logs:"
    echo "  docker compose logs -f"
    echo ""
    echo "To stop the system:"
    echo "  docker compose down"
    echo ""
    echo "To reset all data:"
    echo "  docker compose down -v"
    echo ""
else
    echo ""
    echo "❌ Some services may not have started correctly."
    echo "   Check logs with: docker compose logs"
    exit 1
fi
