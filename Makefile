ctags:
	find . -type f -not -path "*git*" -exec ctags --tag-relative=yes --languages=-javascript {} +
