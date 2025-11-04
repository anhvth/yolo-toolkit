#!/bin/bash
# create_env_labelstudio.sh

ENV_DIR="$HOME/Library/Application Support/label-studio"
ENV_FILE="$ENV_DIR/.env"

# Create directory if missing
mkdir -p "$ENV_DIR"

# Generate random secret key
SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")

# Write .env file
cat > "$ENV_FILE" <<EOF
SECRET_KEY=$SECRET_KEY
DEBUG=True
HOST=localhost
PORT=8080
EOF

echo "✅ .env file created at: $ENV_FILE"
cat "$ENV_FILE"
