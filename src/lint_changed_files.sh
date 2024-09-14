#!/bin/bash

# Get the list of changed Python files
CHANGED_FILES=$(git diff --name-only --diff-filter=ACMRTUXB HEAD~1 HEAD | grep -E '\.py$')

# Check if there are any changed Python files
if [ -z "$CHANGED_FILES" ]; then
  echo "No Python files changed."
  exit 0
fi

# Run Pylint on the changed Python files
echo "Running Pylint on the changed Python files..."
pylint $CHANGED_FILES