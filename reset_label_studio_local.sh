# #!/bin/bash
# # reset_label_studio_local.sh

# # Stop Label Studio if running
# pkill -f "label-studio" 2>/dev/null

# # Remove all Label Studio data
# rm -rf ~/.local/share/label-studio

# # Initialize a fresh install
# label-studio reset_password admin

# # Create admin user
# # label-studio user --username admin --password admin
label-studio start --username admin@example.com --password admin


echo "✅ Reset complete. Admin user: admin@example.com / admin"
