# Prerequisites
* This document is built for lualatex, but should also compile with pdflatex (change in makefile). Fonts will be a bit different then.

# Compiling
* It's easiest to use the makefile, just open a terminal, go to the base folder and type "make" (without quotes, of course). That should call latex several times and other tools (bibtex and makeglossaries) in between. All generated files (including the pdf) will be in build/.
* You can use "make watch" instead to start latexmk in "watch" mode, i.e. compile and recompile whenever you change a file.
* However, the whole process was made for linux. If you are on another operating system, everything except the list of acronyms should still work (maybe without make). Since the list of acronyms is optional, just remove it if you don't want to fiddle with it.
* The manual build process is as usual "pdflatex && bibtex main && pdflatex && pdflatex" which should also work, but without the list of acronyms.

# Misc
* For cleaning, you can call "make clean" or simply delete the build folder.
* Check out the makefile for several fine tunings, e.g. if you want to use pdflatex instead of lualatex, etc.
* Glossaries (acronyms, notation and symbols) are a bit complicated, since we need to call an external program to sort the entries. This is done by latexmk which is configured in .latexmkrc. You should not need to deal with this at all, but in case it fails, you can also try "make glossaries".

