\ProvidesClass{MSthesis} % MSthesis contains everything before "begin document"

\LoadClass[a4paper,12pt,twoside, english]{report} 
% twoside - alternating margins 
% 12pt (same textsize as UiOs word mal)

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{mathptmx}
\usepackage{amsmath}
\usepackage{graphicx}
\newcommand{\minus}{\scalebox{0.75}[1.0]{$-$}}
\usepackage{fancyhdr}
\usepackage{url}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{csquotes}
\usepackage{geometry}
\usepackage{emptypage}
\usepackage[parfill]{parskip} % removes indentation of sectioning.

\linespread{1.5} % Line spacing
\newgeometry{top=2.5cm, bottom=2.5cm, left=3.0cm, right=2.5cm}

%%%  ACRONYMS LIST  %%%%%%%%%%%%%%%%%%%%%%%%
\usepackage[acronym]{glossaries}
\makeglossaries
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Citations  %%%
\usepackage[backend=biber, minbibnames=40, style=authoryear,]{biblatex}
\renewcommand*{\nameyeardelim}{\addcomma\space}


% Include all names of all authors to an article
\ExecuteBibliographyOptions
  {
    maxbibnames = 41,
    minbibnames = 40,
}

% Fix NOU reports
\ifcsundef{abx@macro@\detokenize{date+extrayear}}{}
{
    \renewbibmacro*{date+extrayear}{
        \ifentrytype{online}
        {
            %\addperiod\space     %Gir punktum før årstallet
            \printtext{\printdateextra}
        }
        {
            \iffieldundef{year}{}
            {\printtext[parens]{\printdateextra}}
        }
    }
}
\renewcommand*{\nameyeardelim}{\addcomma\space}

\renewbibmacro*{volume+number+eid}{ %
  \setunit{\addcomma\space}
  \printfield{volume}               %
  %\setunit*{\addnbspace}           % Normal space
  \setunit*{\addnbthinspace}        % Small space
  \printfield{number}%
  \setunit{\addcomma\space}         % 
  \printfield{eid}}
\DeclareFieldFormat[article]{number}{\mkbibparens{#1}}


\setlength\bibitemsep{\baselineskip}
\usepackage{booktabs}
\renewcommand\arraystretch{1.5}

%%%%%% Add you references to the ref.bib file 
\addbibresource{ref.bib}

%\usepackage{subfig}
\usepackage{adjustbox}
\usepackage{subcaption}
\usepackage{wrapfig}
\usepackage{float}
\usepackage{listings}
\usepackage[bottom]{footmisc}
\renewcommand{\vec}[1]{\mathbf{#1}}
\usepackage{xr}
\usepackage{titlesec}
\titleformat{\chapter}[display]   
{\normalfont\huge\bfseries}{\chaptertitlename\ \thechapter}{20pt}{\Huge}   
%\titlespacing*{\chapter}{0pt}{-50pt}{40pt}
\usepackage{pdfpages}
\pagestyle{fancy}
\fancyhf{}


% Page numbering in header 
% RE = right side, even page numbering
% LO = left side, odd page numbering
\fancyfoot[C]{\thepage}
\fancyhead[LE]{\rightmark} % chapter numbering in header 
\fancyhead[RO]{\leftmark} % section numbering in header

% Determines the thickness of the line in the header
% 0pt - removes the line. 
\renewcommand{\headrulewidth}{1pt}
%\renewcommand{\footrulewidth}{1pt}
% Remove this comment if you want i line in the footer. 


\renewcommand\floatpagefraction{.9}
\renewcommand\topfraction{.9}
\renewcommand\bottomfraction{.9}
\renewcommand\textfraction{.1}   
\setcounter{totalnumber}{50}
\setcounter{topnumber}{50}
\setcounter{bottomnumber}{50}
\usepackage[margin=0.8cm,font={small,it}]{caption}

\input{./Frontpage/my_frontpage}

\addbibresource{ref.bib}

