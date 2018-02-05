all: main clean

main:
	pdflatex main.tex
	bibtex main
	pdflatex main.tex
	pdflatex main.tex
	mv main.pdf berger-bhat-report.pdf

clean:
	rm -f *.log *.lof *.lot *.toc *~ *.aux *.fls *.out *.bbl *.blg *.synctex.gz *.maf *.mtc .mtc0 *.snm *.nav *.glo *.glsdefs *.ist *.vrb *.bcf *.run.xml *.glg *.gls *.acn *.acr *.alg

cleanall: clean
	rm -f main.pdf berger-bhat-report.pdf

