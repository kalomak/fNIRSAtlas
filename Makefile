.PHONY: docs docs-gallery docs-clean docs-publish

docs:
	uv run --extra docs sphinx-build -W -j auto -b html docs docs/build/html

docs-gallery:
	uv run --extra docs --extra braindecode --extra dl sphinx-build -W -j auto \
		-D sphinx_gallery_conf.plot_gallery=True \
		-D sphinx_gallery_conf.run_stale_examples=True \
		-b html docs docs/build/html

docs-clean:
	$(MAKE) -C docs clean

docs-publish: docs
	uvx --from ghp-import==2.1.0 ghp-import -n -p -f docs/build/html
