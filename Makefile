PYFILES:=$(shell find . -type f -name '*.py' -not -path '*venv*')
ORGFILES:=$(wildcard docs/*.org)
HTMLFILES:=$(ORGFILES:.org=.html)
IMGFILES:=$(shell find docs -type f -name '*.png' -or -name '*.pdf')

ctags: $(PYFILES)
	find . -type f -not -path "*git*" -exec ctags --tag-relative=yes --languages=-javascript,css,json {} +

etags: $(PYFILES)
	find . -type f -not -path "*git*" -exec ctags -e --tag-relative=yes --languages=-javascript,css,json {} +

fmt: $(PYFILES)
	.venv/bin/black --quiet $<

lint: $(PYFILES)
	.venv/bin/pyflakes $<

data/data.csv: bin/data.py
	.venv/bin/python3 bin/data.py

$(IMGFILES): data/data.csv
	.venv/bin/python3 bin/visualise.py

$(ORGFILES): $(IMGFILES)

%.html: %.org
	rm $@
	emacs $< --batch -f org-html-export-to-html --kill

publish: $(HTMLFILES)

.PHONY: ctags etags fmt
