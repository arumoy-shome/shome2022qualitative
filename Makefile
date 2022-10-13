PYFILES:=$(shell find . -type f -name '*.py' -path '*bin*')
ORGFILES:=$(wildcard docs/*.org)
HTMLFILES:=$(ORGFILES:.org=.html)
IMGFILES:=$(shell find docs -type f -name '*.png' -or -name '*.pdf')

ctags:
	find . -type f -not -path "*git*" -not -path "*vendor*" -exec ctags --tag-relative=yes --languages=-javascript,css,json {} +

etags:
	find . -type f -not -path "*git*" -exec ctags -e --tag-relative=yes --languages=-javascript,css,json {} +

fmt:
	find . -type f -name '*.py' -not -path '*venv*' -not -path '*vendor*' -exec .venv/bin/black {} +

lint:
	find . -type f -name '*.py' -not -path '*venv*' -not -path '*vendor*' -exec .venv/bin/pyflakes {} +

data/data.csv: bin/data.py
	.venv/bin/python3 bin/data.py
	bin/preprocess.bash data/data.csv

visualise: data/data.csv
	.venv/bin/python3 bin/visualise.py

%.html: %.org
	rm $@
	emacs $< --batch -f org-html-export-to-html --kill

data: data/data.csv
publish: $(HTMLFILES)

.PHONY: ctags etags fmt lint data visualise publish
