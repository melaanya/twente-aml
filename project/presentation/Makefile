all: main clean

main:
	xelatex main.tex
	xelatex main.tex
	mv main.pdf berger-bhat-final-presentation.pdf

clean:
	rm -f *.log *.lof *.lot *.toc *~ *.aux *.fls *.out *.bbl *.blg *.synctex.gz *.maf *.mtc .mtc0 *.snm *.nav *.glo *.glsdefs *.ist *.vrb *.bcf *.run.xml *.glg *.gls *.acn *.acr *.alg

cleanall: clean
	rm -f main.pdf berger-bhat-final-presentation.pdf

