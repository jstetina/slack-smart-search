FROM registry.redhat.io/ubi9/python-311:9.6

WORKDIR /app

# Install CPU-only PyTorch first (before sentence-transformers pulls GPU version)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application (MCP server lives in src/)
COPY src/smart_search_mcp.py ./

# Bundle config dir and databases
COPY config/ ./config/
COPY db/ ./db/

# Fix permissions for non-root user
USER 0
RUN chown -R 1001:0 /app/db && chmod -R g+rwX /app/db
USER 1001

# Expose port for HTTP transport
EXPOSE 8000

# Environment variables
ENV MCP_TRANSPORT=http
ENV DB_PATH=/app/db
ENV COLLECTION_NAME=slack_messages
ENV EMBEDDING_MODEL=all-MiniLM-L6-v2
ENV WORKSPACE_URL=https://redhat-internal.slack.com
ENV TOP_K=10

CMD ["python", "smart_search_mcp.py"]
