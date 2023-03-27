ijcai:
	latexmk -pdf -outdir=report report/ijcai23.tex

icse:
	latexmk -pdf -outdir=report report/icse24.tex

present:
	latexmk -xelatex -outdir=report report/presentation.tex

fmt:
	bibtool -s report/report.bib -o report/report.bib

.PHONY: outline report fmt
