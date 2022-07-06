# the main latex file to compile
src ?= main

# where to put compiled files
builddir ?= build
# base directory where the makefile lies in (not in use atm.)
# basedir := $(realpath $(dir $(lastword $(MAKEFILE_LIST))))

# Use latexmk
# This way, you only have to call "make" or "make watch" to build the document.
#
# view=none tells latexmk to not open a viewer, it's the job of "show".
# Alternative: "$pdf_previewer = 'start okular';" in .latexmkrc.
# -lulalatex or -pdflatex tells latexmk to use that engine
cmdEngine ?= latexmk
engineArgs += --view=none
engineArgs += -lualatex
# engineArgs += -pdflatex
#
# If you don't want latexmk, comment cmdEngine and engineArgs above and
# uncomment one of the cmdEngine lines below.
# Remember that you have to call several commands (latex, glossaries, bibtex, latex) now.
# You can use "make manual" for that.
# cmdEngine ?= lualatex
# cmdEngine ?= pdflatex

# more information in logfiles and do not stop and ask on errors
engineArgs += -file-line-error -interaction=nonstopmode
# enable synctex (1: with compression, -1: without comporession)
engineArgs += --synctex=-1
# enable shell escape for e.g. tikz externalize
# engineArgs += -shell-escape

# executed with make show
cmdViewer ?= zathura
# executed with make present (good for presenting beamer slides)
cmdPresenter ?= pdfpc --windowed=both

# uncomment to have this command called when compilation has finished
# BEEP_CMD ?= beep 1



.PHONY: all builddir builddir_only_dir gitignore pdf manual watch bib glossaries show present clean clean_latexmk clean_trash beep


# alias for pdf, this is the default when calling make without a recipe name
all: pdf

builddir:
	@echo "### Creating build dir ###"
	@mkdir -p "$(builddir)"
	@# this is only needed for tikz externalize, but create it always anyways
	@mkdir -p "$(builddir)/$(builddir)"

gitignore:
	@if [ ! -f .gitignore ] ; then \
		echo "### Creating gitignore file. ###" ; \
		echo "$(builddir)/" > .gitignore ; \
	fi


pdf: builddir gitignore
	@echo '### Making pdf ###'
	$(cmdEngine) $(engineArgs) $(PREVIEW_CONTINUOUSLY) --output-directory="$(builddir)" "$(src).tex"
	@$(MAKE) beep

# will compile everything (only makes sense without latexmk)
manual:
	@$(MAKE) pdf
	@$(MAKE) bib
	@$(MAKE) glossaries
	@$(MAKE) pdf
	@$(MAKE) pdf

watch: PREVIEW_CONTINUOUSLY=-pvc
watch: pdf

bib:
	bibtex "$(builddir)/$(src).aux"

glossaries:
	makeglossaries -d "$(builddir)" "$(src)"

show:
	@echo '### Showing pdf ###'
	$(cmdViewer) "$(builddir)/$(src).pdf" >/dev/null 2>&1 &

present:
	@echo '### Presenting pdf ###'
	$(cmdPresenter) "$(builddir)/$(src).pdf" >/dev/null 2>&1 &

clean_trash:
	@echo '### Cleaning build dir with trash ###'
	@(if [ -d "$(builddir)" ]; then \
		trash "$(builddir)"; \
	fi)

clean_latexmk:
	@echo '### Cleaning build dir with latexmk ###'
	latexmk -C --output-directory="$(builddir)"
	@(if [ -d "$(builddir)/$(builddir)" ]; then rmdir "$(builddir)/$(builddir)"; fi)
	@(if [ -d "$(builddir)" ]; then rmdir "$(builddir)"; fi)

# by default, use trash to clean the build dir
clean: clean_trash

beep:
ifdef BEEP_CMD
	@echo "Finished."
	@$(BEEP_CMD)
endif

