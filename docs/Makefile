SPHINXBUILD ?= sphinx-build
SOURCEDIR = .
BUILDDIR = _build
SPHINXOPTS ?= -W --keep-going

.PHONY: html clean doctest linkcheck

html:
	$(SPHINXBUILD) $(SPHINXOPTS) -b html $(SOURCEDIR) $(BUILDDIR)/html

doctest:
	$(SPHINXBUILD) $(SPHINXOPTS) -b doctest $(SOURCEDIR) $(BUILDDIR)/doctest

linkcheck:
	$(SPHINXBUILD) -b linkcheck $(SOURCEDIR) $(BUILDDIR)/linkcheck

clean:
	rm -rf $(BUILDDIR)
