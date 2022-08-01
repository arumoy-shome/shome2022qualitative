PYFILES:=$(shell find . -type f -name '*.py' -not -path '*venv*')

ctags:
	find . -type f -not -path "*git*" -exec ctags --tag-relative=yes --languages=-javascript {} +

etags:
	find . -type f -not -path "*git*" -exec ctags -e --tag-relative=yes --languages=-javascript {} +

fmt: $(PYFILES)
	black $^

lint: $(PYFILES)
	pyflakes $^

.PHONY: ctags etags fmt
