report: 
	latexmk -pdf -outdir=report report/report.tex

present:
	latexmk -xelatex -outdir=report report/presentation.tex

fmt:
	bibtool -s report/report.bib -o report/report.bib

.PHONY: outline report fmt
