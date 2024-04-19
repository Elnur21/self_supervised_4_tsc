#!/bin/bash

# Infinite loop to check for changes every hour
while true; do
    # Change directory to the root of the git repository
    cd "$(git rev-parse --show-toplevel)"

    # Check if there are any changes
    if [ -n "$(git status --porcelain)" ]; then
        echo "Changes detected. Committing and pushing to GitHub..."
        git add .
        git commit -m "Auto commit changes"
        
        # Set credentials for the current push only
        git push -u origin main 
        echo "Changes pushed to GitHub."
        
    else
        echo "No changes detected."
    fi

    # Sleep for an hour before checking again
    sleep 3600  # 3600 seconds = 1 hour
done
