#!/bin/bash
# Create a new user in Label Studio using Python Django shell
# Usage: ./scripts/create_user.sh <username> <password>

if [ $# -ne 2 ]; then
    echo "Usage: $0 <username> <password>"
    echo "Example: $0 hao@example.com 123456"
    exit 1
fi

USERNAME="$1"
PASSWORD="$2"

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_ROOT/label_studio_data"

echo "🔧 Creating user in Label Studio..."
echo "👤 Username: $USERNAME"
echo "📁 Data directory: $DATA_DIR"

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Error: Label Studio data directory not found: $DATA_DIR"
    echo "💡 Start Label Studio first with: python scripts/1_start_labelstudio.py"
    exit 1
fi

# Create Python script to add user via Django shell
cd "$PROJECT_ROOT"

python -c "
import os
import sys
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'label_studio.core.settings.label_studio')
os.environ['LABEL_STUDIO_BASE_DATA_DIR'] = '$DATA_DIR'

# Suppress verbose output
import warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# Setup Django
sys.path.insert(0, '$(python -c "import label_studio; import os; print(os.path.dirname(label_studio.__file__))")')
django.setup()

from django.contrib.auth import get_user_model
User = get_user_model()

try:
    user, created = User.objects.get_or_create(
        email='$USERNAME',
        defaults={'username': '$USERNAME'}
    )
    if created:
        user.set_password('$PASSWORD')
        user.save()
        print('✅ User created successfully!')
    else:
        # Update password if user exists
        user.set_password('$PASSWORD')
        user.save()
        print('✅ User already exists - password updated!')
    print('👤 Username: $USERNAME')
    print('🔑 Password: $PASSWORD')
    print('🔗 Login at: http://localhost:8080')
except Exception as e:
    print(f'❌ Error: {e}')
    sys.exit(1)
" 2>&1 | grep -v "Provider\|Looking for locale\|Not in REPL\|NumExpr\|FutureWarning\|locale.*is not\|get 'SECRET_KEY'\|Database and media\|Static URL\|Read environment" | grep -v "^$"

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    exit 0
else
    echo "❌ Failed to create user"
    exit 1
fi
