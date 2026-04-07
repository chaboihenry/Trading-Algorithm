#!/bin/bash

echo "Committing updated models to GitHub..."
git add the_models/
git commit -m "AUTO: M1 Payload Update (Universe & ML Models)"
git push origin main
echo "Push successful."