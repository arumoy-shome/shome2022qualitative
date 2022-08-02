PYFILES:=$(shell find . -type f -name '*.py' -not -path '*venv*')

ctags:
	find . -type f -not -path "*git*" -exec ctags --tag-relative=yes --languages=-javascript {} +

etags:
	find . -type f -not -path "*git*" -exec ctags -e --tag-relative=yes --languages=-javascript {} +

fmt: $(PYFILES)
	.venv/bin/black $^

lint: $(PYFILES)
	.venv/bin/pyflakes $^

.PHONY: ctags etags fmt
