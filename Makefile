PYFILES:=$(shell find . -type f -name '*.py' -path '*bin*')
ORGFILES:=$(wildcard docs/*.org)
HTMLFILES:=$(ORGFILES:.org=.html)
IMGFILES:=$(shell find docs -type f -name '*.png' -or -name '*.pdf')

ctags:
	find . -type f -not -path "*git*" -not -path "*vendor*" -exec ctags --tag-relative=yes --languages=-javascript,css,json {} +

etags:
	find . -type f -not -path "*git*" -not -path "*vendor*" -exec ctags -e --tag-relative=yes --languages=-javascript,css,json {} +

fmt:
	find . -type f -name '*.py' -not -path '*venv*' -not -path '*vendor*' -exec black {} +

lint:
	find . -type f -name '*.py' -not -path '*venv*' -not -path '*vendor*' -exec pyflakes {} +

.PHONY: ctags etags fmt lint
