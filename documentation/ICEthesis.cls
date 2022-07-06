% 2021 Institute for communications engineering (ICE) Thesis class based on the 2008 LNT thesis class
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{ICEthesis}[2021/11/10 ICE Thesis class]

% option parser
\RequirePackage{kvoptions}
\SetupKeyvalOptions{family=ICEthesis,prefix=ICEthesis@}

% selects the chair, default is lnt
\DeclareStringOption[lnt]{chair}

\def\ICEthesissetup#1{\kvsetkeys{ICEthesis}{#1}}

% Pass unknown options to scrreprt class
\DeclareDefaultOption{\PassOptionsToClass{\CurrentOption}{scrreprt}}
% Process options (MUST BE BEFORE LOADCLASS COMMAND!)
\ProcessKeyvalOptions*
\PassOptionsToClass{%
    a4paper, % use A4
    DIV=12, % document layout: DIV factor = 12 (larger number creates larger pages)
    BCOR=12mm, % binding correction: 12mm
    headsepline, % separate page head by a line
    twoside, % twosided document
    11pt, % font size
    bibliography=totoc, % show bibliography in the table of contents
    openright, % start new chapters only on right pages (odd numbered pages)
    cleardoublepage=empty, % start new chapters on the right page
}{scrreprt}

% This class is built upon the KOMAscript scrreprt class
\LoadClass{scrreprt}

% Load necessary packages
\RequirePackage[english]{babel} % english language
\RequirePackage{graphicx}       % for using eps images
\RequirePackage{verbatim}       % for quickly commenting out large parts of your text
\RequirePackage{amssymb}        % additional math symbols
\RequirePackage{amsmath}        % additional math commands
\RequirePackage{amsxtra}        % even more math symbols
\RequirePackage{amsthm}         % theorem environment etc
\RequirePackage{setspace}		% set line spacing globally
% more information on amsmath: http://www.ctan.org/get/macros/latex/required/amslatex/math/amsldoc.pdf
\RequirePackage{psfrag}         % psfrag: http://www.ctan.org/get/macros/latex/contrib/psfrag/pfgguide.pdf
\RequirePackage{subfigure}      % enable subfigures

\RequirePackage{tabularx} % flexible tabulars



% Define strings depending on the selected chairs
\def\ICEthesis@chairname@lnt{Lehrstuhl für Nachrichtentechnik}
\def\ICEthesis@chairname@cod{Professur für Coding and Cryptography}
\def\ICEthesis@chairname@lut{Professur für Leitungsgebundene Übertragungstechnik}

\def\ICEthesis@chairnameShort@lnt{LNT}
\def\ICEthesis@chairnameShort@cod{COD}
\def\ICEthesis@chairnameShort@lut{LÜT}

\def\ICEthesis@amAnder@lnt{am}
\def\ICEthesis@amAnder@cod{an der}
\def\ICEthesis@amAnder@lut{an der}

\def\ICEthesis@profname@lnt{Prof.~Dr.sc.tech.~Gerhard~Kramer}
\def\ICEthesis@profname@cod{Prof.~Dr.-Ing.~Antonia~Wachter-Zeh}
\def\ICEthesis@profname@lut{Prof.~Dr.-Ing.~Norbert~Hanik}

\def\ICEthesis@header@lnt{
\begin{tabular}[b]{@{}l@{}}%
    \textsc{Technische Universität München}\\
    \textsc{Lehrstuhl für Nachrichtentechnik}\\
    \ICEthesis@printProfname%
\end{tabular}%
}
\def\ICEthesis@header@cod{
\begin{tabular}[b]{@{}l@{}}%
    \textsc{Technische Universität München}\\
    \textsc{Professur für Coding and Cryptography}\\
    \ICEthesis@printProfname%
\end{tabular}%
}
\def\ICEthesis@header@lut{
\begin{tabular}[b]{@{}l@{}}%
    \textsc{Technische Universität München}\\
    \textsc{Professur für}\\
    \textsc{Leitungsgebundene Übertragungstechnik}\\
    \ICEthesis@printProfname%
\end{tabular}%
}

% define commands which will print the selected strings depending on the selected chair
\def\ICEthesis@printChairname{\csname ICEthesis@chairname@\ICEthesis@chair \endcsname}
\def\ICEthesis@printChairnameShort{\csname ICEthesis@chairnameShort@\ICEthesis@chair \endcsname}
\def\ICEthesis@printAmAnder{\csname ICEthesis@amAnder@\ICEthesis@chair \endcsname}
\def\ICEthesis@printProfname{\csname ICEthesis@profname@\ICEthesis@chair \endcsname}
\def\ICEthesis@printHeader{\csname ICEthesis@header@\ICEthesis@chair \endcsname}


% Set chapter, section and subsection to regular LaTeX font
\addtokomafont{sectioning}{\rmfamily}
\addtokomafont{descriptionlabel}{\rmfamily}

% Set general pagesyle
\pagestyle{headings}

% Revome paragraph indent
\parindent 0em

% Set line spacing globally
\onehalfspacing

% Set depth of table of contents to 2 (this includes sections and subsections)
\setcounter{tocdepth}{2}


% \renewcommand\tabularxcolumn[1]{m{#1}}% for vertical alighnment
% Title page
\newcommand{\ICEtitle}[6]{%
    \begingroup%
    \titlehead{%
        \vspace*{-40pt}%
        \begin{tabularx}{\textwidth}{Xr}%
            {\ICEthesis@printHeader{}} & \includegraphics[height=16.5mm]{TUMlogo_tumcd}%
        \end{tabularx}%
        \vspace*{1ex}%
        \hrule%
    }
    \subject{\vspace*{2.5cm}#1}%
    \title{#2}%
    \author{\vspace*{1cm}\\Vorgelegt von:\\#3}%
    \publishers{\vspace*{2.5cm}Betreut von:\\#4\\[-2cm]}%
    \uppertitleback{#1 \ICEthesis@printAmAnder\\%
                \ICEthesis@printChairname{} (\ICEthesis@printChairnameShort)\\%
                der Technischen Universität München (TUM)\\%
                Titel : #2\\%
                Autor : #3}%
    \lowertitleback{#6}%
    \date{#5}%
    \maketitle%
    \endgroup%
}



% *******************************************************
% Rechtliche Erklärung
\newcommand{\ICErecht}[2]{
	\begingroup
	\onehalfspacing
    \thispagestyle{empty}
    \
    \vfill
    Ich versichere hiermit wahrheitsgemäß, die Arbeit bis auf die dem Aufgabensteller bereits bekannte Hilfe selbständig angefertigt, alle benutzten Hilfsmittel vollständig und genau angegeben und alles kenntlich gemacht zu haben, was aus Arbeiten anderer unverändert oder mit Abänderung entnommen wurde.\\
    \begin{center}
    \raggedright #2\\
    \vspace*{-2ex}
    \dotfill\\
    Ort, Datum \hfill (#1)\\
    \end{center}
    \endgroup
}
