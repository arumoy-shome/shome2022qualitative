PYFILES:=$(shell find . -type f -name '*.py' -not -path '*venv*')

ctags: $(PYFILES)
	find . -type f -not -path "*git*" -exec ctags --tag-relative=yes --languages=-javascript,css,json {} +

etags: $(PYFILES)
	find . -type f -not -path "*git*" -exec ctags -e --tag-relative=yes --languages=-javascript,css,json {} +

fmt: $(PYFILES)
	.venv/bin/black --quiet $<

lint: $(PYFILES)
	.venv/bin/pyflakes $<

.PHONY: ctags etags fmt
