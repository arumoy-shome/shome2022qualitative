outline:
	fd -e=md -e=tex -X ctags --tag-relative=yes {}

report: 
	latexmk -pdf -outdir=report report/report.tex

.PHONY: outline report
