run:
	@python3 main.py

debug:
	@python3 -m pdb main.py

test:
	@python3 test.py

time:
	@python3 time_testing.py

sprawko: latex-compile
	sioyek ./sprawozdanie/sprawozdanie.pdf

latex-compile:
	pdflatex -output-directory="./sprawozdanie" ./sprawozdanie.tex

