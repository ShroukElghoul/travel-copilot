# Base image: official Python 3.11, slim variant (Debian without dev tools).
# Keeps the image small — no compilers, no package managers we don't need.
FROM python:3.11-slim

# All subsequent commands run from /app inside the container.
WORKDIR /app

# Copy requirements first — before the code.
# Docker builds in layers. If requirements.txt hasn't changed, Docker reuses
# the cached pip install layer and skips it on the next build. Fast rebuilds.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the project code.
COPY src/ ./src/
COPY scripts/ ./scripts/

# Document that the container listens on port 8000.
# This doesn't actually open the port — that happens at `docker run` time.
EXPOSE 8000

# The command that runs when the container starts.
# --host 0.0.0.0 is critical: without it Uvicorn binds to 127.0.0.1 (loopback
# only) and the container is unreachable from outside.
CMD ["uvicorn", "src.travel_copilot.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
