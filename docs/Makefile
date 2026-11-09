SPHINXOPTS    ?= -j auto
SPHINXBUILD   = sphinx-build
SOURCEDIR     = .
BUILDDIR      = build

.PHONY: clean html

clean:
	-rm -rf $(BUILDDIR) generated auto_examples sg_execution_times.rst

html:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
