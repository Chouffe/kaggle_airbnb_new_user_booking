#!/bin/bash

# Generating pdf from ipynb
~/anaconda2/bin/ipython nbconvert --to pdf ../notebooks/Capstone*.ipynb
mv Capstone*.pdf analysis.pdf

# Generating pdf from markdown file
pandoc report.md -s -o report.pdf

# Merging the two pdfs in a report called capstone.pdf
pdftk A=report.pdf B=analysis.pdf cat A1-4 B A6-end output capstone.pdf

# Removes tmp files
rm analysis.pdf report.pdf
