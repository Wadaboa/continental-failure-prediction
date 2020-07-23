# LaTeX Beamer template
This repository contains a template to be used as a starting point to build eye-catching presentations.\
The template is written in `LaTeX`, using the well-known `Beamer` class.\
The theme used is [`Metropolis`](https://github.com/matze/mtheme), which is available on `CTAN` 
and already shipped with the latest versions of the TeX distributions. 

## Other packages
This template uses the following additional packages:
* `minted`, which is on `CTAN`, but has the additional dependency of `pygments`, which is a `Python` package that could be installed usig pip

## Notes
In order to correctly compile this template, you should use `XeLaTeX` with the following command-line arguments:
```bash
-xelatex -synctex=1 -interaction=nonstopmode -file-line-error -shell-escape -8bit %DOC%
```
If you want to have a look at the template, open the file [`template.pdf`](Slides/template.pdf)
