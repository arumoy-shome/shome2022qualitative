ctags:
	find . -type f -not -path "*git*" -exec ctags --tag-relative=yes --languages=-javascript {} +

etags:
	find . -type f -not -path "*git*" -exec ctags -e --tag-relative=yes --languages=-javascript {} +
