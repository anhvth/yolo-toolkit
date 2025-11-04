#!/usr/bin/env python3
"""
Script 1: Start Label Studio Server
This script launches Label Studio locally (for non-Docker setups).
For Docker users, use: docker compose up
"""

import os
import sys

def start_labelstudio():
    """Start Label Studio server"""
    print("🚀 Starting Label Studio server...")
    print("Note: If using Docker, run 'docker compose up' instead")
    print("Access Label Studio at: http://localhost:8080")
    print("\nPress Ctrl+C to stop the server\n")
    
    # Start label-studio
    os.system("label-studio start")

if __name__ == "__main__":
    try:
        start_labelstudio()
    except KeyboardInterrupt:
        print("\n\n✅ Label Studio server stopped")
        sys.exit(0)
