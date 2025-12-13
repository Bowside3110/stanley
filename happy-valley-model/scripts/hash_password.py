#!/usr/bin/env python3
"""
Utility script to hash passwords for the Stanley Racing web application.
Usage: python scripts/hash_password.py <password>
"""

from passlib.context import CryptContext
import sys

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def main():
    if len(sys.argv) > 1:
        password = sys.argv[1]
        hashed = pwd_context.hash(password)
        print(f"\nHashed password: {hashed}\n")
        print("Copy this hash and update the USERS dict in web/auth.py")
    else:
        print("Usage: python scripts/hash_password.py <password>")
        print("\nExample:")
        print("  python scripts/hash_password.py mySecurePassword123")
        sys.exit(1)

if __name__ == "__main__":
    main()

