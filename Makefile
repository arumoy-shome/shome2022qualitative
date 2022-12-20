outline:
	fd -e=md -e=tex -X ctags --tag-relative=yes {}

report: 
	latexmk -pdf -outdir=report report/report.tex

present:
	latexmk -xelatex -outdir=report report/presentation.tex

.PHONY: outline report present
