set windows-shell := ["cmd.exe", "/c"]

APP_NAME := "quantum"
SRC_DIR := "src"
DIST_DIR := "dist"
BUILD_DIR := "build"
TESTS_DIR := ""

# Default recipe - show help
default:
    @just --list

# Install the application globally on the system
install-dev:
    @uv tool install -e ".[cli]" -n

# =================================
# === Build Executable Commands ===
# =================================

# Build the executable
build:
    uv build --wheel

# Clean up build and distribution directories
[windows]
clean:
    rd /s /q {{DIST_DIR}} {{BUILD_DIR}} 2>nul

# Clean up build and distribution directories
[unix]
clean:
    rm -rf {{DIST_DIR}} {{BUILD_DIR}} 2>/dev/null


# =============================
# === Code Quality Commands ===
# =============================

# Run type checking
ty:
   @uv run ty check {{SRC_DIR}} {{TESTS_DIR}}

# Run linting
check: ty
    @uv run ruff check {{SRC_DIR}} {{TESTS_DIR}}

# Run linting with auto-fix
check-fix:
    @uv run ruff check {{SRC_DIR}} {{TESTS_DIR}} --fix

# Check code formatting without making changes
format-diff:
    uv run ruff format --diff {{SRC_DIR}} {{TESTS_DIR}}

# Format code
format: check-fix
    uv run ruff format {{SRC_DIR}} {{TESTS_DIR}}

# Run all code quality checks
check-all: check format-diff
