#!/bin/bash

SOURCE=bayespy
TARGET=bayespy-arxiv
SUPP='jmlr2e.sty'

# Remove comments with:
perl -pe 's/(^|[^\\])%.*/\1%/' < $SOURCE.tex > tmp.tex 

# To use pdflatex, add the following to first five lines:
(head -n 1 tmp.tex; echo \\pdfoutput=1; tail -n +2 tmp.tex) > $TARGET.tex

# Compile and make package
pdflatex $TARGET && bibtex $TARGET && pdflatex $TARGET && pdflatex $TARGET && tar -czf $TARGET.tar.gz $TARGET.tex $TARGET.bbl $SUPP

