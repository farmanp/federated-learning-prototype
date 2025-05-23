#!/bin/bash

# Federated Learning Docker Management Script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
}

# Function to build images
build_images() {
    print_status "Building Docker images..."
    cd "$PROJECT_ROOT"
    
    print_status "Building aggregator image..."
    docker build -f docker/Dockerfile.aggregator -t fl-aggregator .
    
    print_status "Building data party image..."
    docker build -f docker/Dockerfile.data_party -t fl-data-party .
    
    print_status "Images built successfully!"
}

# Function to start the federated learning system
start_system() {
    print_status "Starting federated learning system..."
    cd "$PROJECT_ROOT/docker"
    
    # Create necessary directories
    mkdir -p ../logs ../data
    
    # Start with docker-compose
    docker-compose up -d
    
    print_status "System started! Check logs with: $0 logs"
    print_status "Aggregator available at: localhost:50051"
    print_status "Data parties available at: localhost:50052-50054"
}

# Function to stop the system
stop_system() {
    print_status "Stopping federated learning system..."
    cd "$PROJECT_ROOT/docker"
    docker-compose down
    print_status "System stopped!"
}

# Function to show logs
show_logs() {
    cd "$PROJECT_ROOT/docker"
    if [ -z "$2" ]; then
        docker-compose logs -f
    else
        docker-compose logs -f "$2"
    fi
}

# Function to scale data parties
scale_parties() {
    local num_parties=${2:-3}
    print_status "Scaling to $num_parties data parties..."
    cd "$PROJECT_ROOT/docker"
    
    # Generate additional config files if needed
    for i in $(seq 4 $num_parties); do
        if [ ! -f "../config/party_$i.yaml" ]; then
            sed "s/party_1/party_$i/g; s/id: 1/id: $i/g; s/partition_index: 0/partition_index: $((i-1))/g" ../config/party_1.yaml > "../config/party_$i.yaml"
        fi
    done
    
    docker-compose up -d --scale data_party_1=$num_parties
    print_status "Scaled to $num_parties parties!"
}

# Function to clean up
cleanup() {
    print_status "Cleaning up Docker resources..."
    cd "$PROJECT_ROOT/docker"
    docker-compose down -v --remove-orphans
    docker system prune -f
    print_status "Cleanup complete!"
}

# Function to show status
show_status() {
    print_status "Federated Learning System Status:"
    cd "$PROJECT_ROOT/docker"
    docker-compose ps
}

# Function to run tests
run_tests() {
    print_status "Running tests in Docker environment..."
    cd "$PROJECT_ROOT"
    docker run --rm -v "$PWD:/app" -w /app fl-aggregator uv run pytest tests/
}

# Main script logic
case "$1" in
    build)
        check_docker
        build_images
        ;;
    start)
        check_docker
        build_images
        start_system
        ;;
    stop)
        check_docker
        stop_system
        ;;
    restart)
        check_docker
        stop_system
        build_images
        start_system
        ;;
    logs)
        check_docker
        show_logs "$@"
        ;;
    scale)
        check_docker
        scale_parties "$@"
        ;;
    status)
        check_docker
        show_status
        ;;
    test)
        check_docker
        run_tests
        ;;
    clean)
        check_docker
        cleanup
        ;;
    *)
        echo "Usage: $0 {build|start|stop|restart|logs|scale|status|test|clean}"
        echo ""
        echo "Commands:"
        echo "  build     - Build Docker images"
        echo "  start     - Build and start the system"
        echo "  stop      - Stop the system"
        echo "  restart   - Restart the system"
        echo "  logs      - Show logs (optionally specify service name)"
        echo "  scale N   - Scale to N data parties"
        echo "  status    - Show system status"
        echo "  test      - Run tests"
        echo "  clean     - Clean up Docker resources"
        echo ""
        echo "Examples:"
        echo "  $0 start"
        echo "  $0 logs aggregator"
        echo "  $0 scale 5"
        exit 1
        ;;
esac
