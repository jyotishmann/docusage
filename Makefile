# Makefile -- DocuSage development workflow
.PHONY: install test test-fast test-cov lint type-check clean index run ci

# ---- Environment --------------------------------------------------------
install:
    pip install -r requirements.txt --break-system-packages

# ---- Testing ------------------------------------------------------------
test:
    pytest tests/ -v --tb=short

test-fast:
    pytest tests/ -v --tb=short -m "not slow"

test-cov:
    pytest tests/ -v --tb=short -m "not slow" \
      --cov=. --cov-report=term-missing --cov-omit="tests/*,scripts/*"

ci: test-fast

# ---- Code quality -------------------------------------------------------
lint:
    ruff check . --select E,W,F,I --ignore E501

type-check:
    mypy corpus/ indexing/ retrieval/ query/ generation/ audit/ pipeline/ \
      --ignore-missing-imports --no-strict-optional

format:
    ruff format .

# ---- Corpus and index ---------------------------------------------------
download-corpus:
    python scripts/download_corpus.py --phase 1

build-index:
    python scripts/build_index.py

index: download-corpus build-index

# ---- Application --------------------------------------------------------
run:
    python app.py

# ---- Cleanup ------------------------------------------------------------
clean:
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete 2>/dev/null || true
    find . -name ".coverage" -delete 2>/dev/null || true
    find . -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true