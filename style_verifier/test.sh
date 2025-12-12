#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "Generating 10 samples..."
rm -f outputs.txt

for i in {1..10}; do
    echo "Sample $i..."
    python generate.py >> outputs.txt
    echo "---" >> outputs.txt
done

echo ""
echo "Checking for banned phrases..."
echo ""

if grep -i -E "here's what|is dead|are dead|real unlock|real secret|I was wrong|Let me explain|In this article|In this thread|I used to think|But here's|And here" outputs.txt; then
    echo ""
    echo "FAIL: Found banned phrases in output!"
    exit 1
else
    echo "PASS: No banned phrases found."
fi
