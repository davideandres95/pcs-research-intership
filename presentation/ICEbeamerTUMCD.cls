%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Provides a template class for presentations                 %
% Mostly a collection of the official template into a class   %
% Author: Lukas Holzbaur                                      %
% Date: 29.04.2020                                            %
% Last change: Lukas Holzbaur - 29.04.2020                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Identification
\NeedsTeXFormat{LaTeX2e}
\ProvidesPackage{ICEbeamerTUMCD}[2019/10/27 ICE Beamer Class]

\typeout{}
\typeout{-------------------------------------------------------------------}
\typeout{ICEbeamer TUM CD Class (ICEbeamerTUMCD.sty) Version 0.92, Apr. 2020}
\typeout{           Institute for Communications Engineering}
\typeout{           Technische Universitaet Muenchen}
\typeout{-------------------------------------------------------------------}

% Class options

\RequirePackage{kvoptions}%

\newif\if@ICEGermanLanguage
\@ICEGermanLanguagefalse
\DeclareOption{german}{  \PassOptionsToPackage{ngerman}{babel} \@ICEGermanLanguagetrue}
\DeclareOption{ngerman}{ \PassOptionsToPackage{ngerman}{babel} \@ICEGermanLanguagetrue}
\DeclareOption{english}{ \PassOptionsToPackage{english}{babel} \@ICEGermanLanguagefalse}%
\DeclareOption*{ \PackageWarning{ICEbeamer}{Unknown option ‘\CurrentOption’} }

\ExecuteOptions{english}



% This option enables the official TUM math fonts (looks like ****)
\newif\ifUseMathTUMCD
\UseMathTUMCDfalse
\DeclareOption{mathTUMCD}{\UseMathTUMCDtrue}

% Enables Helvetica font
\newif\ifuseHelvet
\useHelvetfalse
\DeclareOption{helvet}{\useHelvettrue}

% Enables nice title fonts (textrm)
\newif\ifuseNiceTitles
\useNiceTitlesfalse
\DeclareOption{nicetitles}{\useNiceTitlestrue}

% Handout options
\newif\if@notes
\@notesfalse
\DeclareOption{notes}{\@notestrue}

\newif\if@ruled
\@ruledfalse
\DeclareOption{ruled}{\@ruledtrue}

% Set aspect ratio
% \DeclareOption{1610}{\edef\scalingFactorNotesPosition{0.3} \PassOptionsToPackage{orientation=landscape,
%     size=custom,
%     width=25.4,
%     height=15.875,
%     scale=0.6}{beamerposter}}
% \DeclareOption{169}{\edef\scalingFactorNotesPosition{0.35} \PassOptionsToPackage{orientation=landscape,
%     size=custom,
%     width=25.4,
%     height=14.2875, %16:9
%     scale=0.6}{beamerposter}}
% \DeclareOption{43}{\edef\scalingFactorNotesPosition{0.22} \PassOptionsToPackage{orientation=landscape,
%     size=custom,
%     width=25.4,
%     height=19.05,
%     scale=0.6}{beamerposter}}
\DeclareOption*{ \PackageWarning{ICEbeamer}{Unknown option ‘\CurrentOption’}
\PassOptionsToClass{\CurrentOption}{beamer}}


\ProcessOptions\relax

\LoadClass{beamer}

% \RequirePackage{beamerposter}


% Language settings:
\if@ICEGermanLanguage
\typeout{Setting German language...}
\RequirePackage[ngerman]{babel}
\RequirePackage[utf8]{inputenc}	% Allows to directly enter Umlaute in .tex files
\RequirePackage[T1]{fontenc}	% Allows to use correct hyphenation with Umlaute
\newcommand{\@ICElanguagestringTUM}{Technische Universit\"at M\"unchen}
\newcommand{\@ICElanguagestringEI}{Fakultät für Elektrotechnik und Informationstechnik}
\newcommand{\@ICElanguagestringICE}{Institute for Communications Engineering}
\else
\typeout{Setting English language...}
\RequirePackage[english]{babel}
\newcommand{\@ICElanguagestringTUM}{Technical University of Munich}
\newcommand{\@ICElanguagestringEI}{Department of Electrical and Computer Engineering}
\newcommand{\@ICElanguagestringICE}{Institute for Communications Engineering}
\fi


% Choose Font
\ifuseHelvet
% Helvetica, used by TUM CD templates
\RequirePackage[scaled]{helvet}
\renewcommand{\familydefault}{\sfdefault}
\else
\RequirePackage{lmodern}	% Uses Postscript fonts which greatly improves visual quality and file size!
\fi




% Math font setting
\ifUseMathTUMCD
\typeout{Using TUMCD math font}
\RequirePackage{mathptmx}
\fi




\newcommand{\PraesentationAbstandAbsatz}{18pt}
\newcommand{\PraesentationPositionKorrekturOben}{-2.65cm}

\RequirePackage[utf8]{inputenc}

\RequirePackage{calc} % Berechnungen

% \RequirePackage[ngerman]{babel} % Deutsche Lokalisierung
\RequirePackage{graphicx} % Grafiken
\RequirePackage[export]{adjustbox}
\RequirePackage[absolute, overlay]{textpos} % Positionierung

% Silbentrennung:
\RequirePackage{hyphenat}
% \tolerance 2414
% \hbadness 2414
% \emergencystretch 1.5em
% \hfuzz 0.3pt
% \widowpenalty=10000     % Hurenkinder
% \clubpenalty=10000      % Schusterjungen
% \vfuzz \hfuzz

% Euro-Symbol:
\RequirePackage[gen]{eurosym}
\DeclareUnicodeCharacter{20AC}{\euro{}}


% Uncomment the following to use official TUM math fonts (looks like ****)
% \RequirePackage{mathptmx} % skalierbare Formelschriften

\RequirePackage{tabularx}

\RequirePackage{multicol} % mehrspaltiger Text

\RequirePackage{tikz}
\usetikzlibrary{arrows, shapes, shapes.multipart, trees, positioning,
  backgrounds, fit, matrix}

% Diagramme:
\RequirePackage{pgfplots}
\pgfplotsset{compat=default}

% Erweiterbare Fusszeile:
\newcommand{\PraesentationFusszeileZusatz}{}



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EINSTELLUNGEN
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% input{./Ressourcen/_Informationen.tex}
% Allgemein:
\newcommand{\AllgemeinGestalter}{ediundsepp Gestaltungsgesellschaft}
\newcommand{\AllgemeinErsteller}{eWorks GmbH}

% Universität:
\newcommand{\UniversitaetName}{Technische Universität München}
\newcommand{\UniversitaetAbkuerzung}{TUM}
\newcommand{\UniversitaetWebseite}{www.tum.de}
\newcommand{\UniversitaetLogoBreite}{19mm}
\newcommand{\UniversitaetLogoHoehe}{1cm}

\newcommand{\UniversitaetAdresse}{%
  Arcisstraße~21\\%
  80333~München%
}
%%% Ende  _Informationen


\RequirePackage{xparse}
\ExplSyntaxOn
\cs_new_eq:NN \calc \fp_eval:n
\ExplSyntaxOff

% Make sure that logo size and border are constant for any aspect ratio
% 455.24 is width of the 169 and 1610 aspect ratio - used as the default
% Normalized papersizes - useful for making things same size for all aspect ratios
\newlength{\normPaperwidth} 
\setlength{\normPaperwidth}{\calc{(455.24/\paperwidth)} \paperwidth}
\newlength{\normPaperheight} 
\setlength{\normPaperheight}{\calc{(256/\paperheight)} \paperheight}

% Width of TUM logo
\newlength{\logowidth}
\setlength{\logowidth}{0.066 \normPaperwidth}

\newcommand{\PraesentationSeitenrand}{0.035 \normPaperwidth}
\newcommand\crule[3][black]{\textcolor{#1}{\rule{#2}{#3}}}

% Space between title and header
\newlength{\offsetTitle}
\setlength{\offsetTitle}{0.15 \paperheight}%

% Space between title and author
\newlength{\authorOffsetTitlepage}
\setlength{\authorOffsetTitlepage}{0.14 \paperheight}


\subtitle{\phantom{Placeholder}}

\newlength\smallerbaselineskip
\setlength{\smallerbaselineskip}{0.8\baselineskip}

% Define TUM colors

\RequirePackage{xcolor}
% Blautöne:
\definecolor{TUMBlau}{RGB}{0,101,189} % Pantone 300 (0, 0.396, 0.7412)
\definecolor{TUMBlauDunkel}{RGB}{0,82,147} % Pantone 301 (0, 0.3216,0.5765)
\definecolor{TUMBlauHell}{RGB}{152,198,234} % Pantone 283 (0.596,0.776,0.918)
\definecolor{TUMBlauMittel}{RGB}{100,160,200} % Pantone 542 (0.392, 0.627, 0.784)

% Hervorhebung:
\definecolor{TUMElfenbein}{RGB}{218,215,203} % Pantone 7527 -Elfenbein
\definecolor{TUMGruen}{RGB}{162,173,0} % Pantone 383 - Grün (0.6353,0.6784,0)
\definecolor{TUMOrange}{RGB}{227,114,34} % Pantone 158 - Orange (0.8902, 0.4471, 0.133)
\definecolor{TUMGrau}{gray}{0.6} % Grau 60%

% More colors, not sure if super official
\definecolor{TUMGruenDunkel}{RGB}{0,124,48} % (0,0.4863,0.1882)
\definecolor{TUMRot}{RGB}{196,7,27} % (0.7686,0.02745,0.10588)

% English colors
\definecolor{TUMBlue}{RGB}{0,101,189} % Pantone 300 (0, 0.396, 0.7412)
\definecolor{TUMBlueDark}{RGB}{0,82,147} % Pantone 301 (0, 0.3216,0.5765)
\definecolor{TUMBlueLight}{RGB}{152,198,234} % Pantone 283 (0.596,0.776,0.918)
\definecolor{TUMBlueMedium}{RGB}{100,160,200} % Pantone 542 (0.392, 0.627, 0.784)

% Hervorhebung:
\definecolor{TUMIvory}{RGB}{218,215,203} % Pantone 7527 -Elfenbein
\definecolor{TUMGreen}{RGB}{162,173,0} % Pantone 383 - Grün (0.6353,0.6784,0)
\definecolor{TUMGray}{gray}{0.6} % Grau 60%

% More colors, not sure if super official
\definecolor{TUMGreenDark}{RGB}{0,124,48} % (0,0.4863,0.1882)
\definecolor{TUMRed}{RGB}{196,7,27} % (0.7686,0.02745,0.10588)

\setbeamercolor*{alerted text}{fg=TUMOrange}

\newcommand{\PraesentationSetzeTextfarbe}{%
  \color{PraesentationTextfarbe}%
  \setbeamercolor*{frametitle}{fg=PraesentationTextfarbe}%
  \setbeamercolor*{normal text}{fg=PraesentationTextfarbe}%
  \setbeamercolor{itemize/enumerate body}{fg=PraesentationTextfarbe}%
  \setbeamercolor*{itemize item}{fg=PraesentationTextfarbe}%
}

\newcommand{\PraesentationFarbschemaStandard}{%
  \setbeamercolor*{background canvas}{}%
  \definecolor{PraesentationTextfarbe}{rgb}{0,0,0}%
  \PraesentationSetzeTextfarbe%
}

\newcommand{\PraesentationTitelseiteInhalt}{%
  % \begin{textblock*}{\paperwidth}[0,0](0cm,-\PraesentationSeitenrand - 6.5mm + \PraesentationPositionKorrekturOben)%
    % \color{PraesentationTextfarbe}%
    % \frametitle{\\[4ex] \huge \inserttitle}
    \vspace*{\offsetTitle}%
    \ifuseNiceTitles
    % Use rm font for titles
    {\fontsize{24}{32} \selectfont \textrm \inserttitle}\\[0.9em]
    {\fontsize{16}{32} \selectfont \textrm \insertsubtitle}\\
    % \vspace*{95mm}%
    \vspace*{\authorOffsetTitlepage}%
    {\fontsize{10}{12} \selectfont \textrm \insertauthor}\\[0.3em]%
    {\fontsize{10}{12} \selectfont \textrm \insertinstitute}\\%
    {\fontsize{10}{12} \selectfont \textrm \Datum}
    \else
    {\fontsize{24}{32} \selectfont \inserttitle}\\[0.9em]
    {\fontsize{16}{32} \selectfont \insertsubtitle}\\
    % \vspace*{95mm}%
    \vspace*{\authorOffsetTitlepage}%
    {\fontsize{10}{12} \selectfont \insertauthor}\\[0.3em]%
    {\fontsize{10}{12} \selectfont \insertinstitute}\\%
    {\fontsize{10}{12} \selectfont \Datum}
    \fi
    % \insertdate%
  % \end{textblock*}%
}

\newcommand{\PraesentationSeitenkopfInhalt}[1]{%
  % \vspace*{31.7mm}%
  \begin{textblock*}{\logowidth}[1,0](\paperwidth - \PraesentationSeitenrand - \PraesentationSeitenrand, 0cm)%
    \includegraphics[width= \logowidth]{#1}%
  \end{textblock*}%
  \begin{textblock*}{0.118 \normPaperwidth}[1,0](\paperwidth - \PraesentationSeitenrand, -\PraesentationSeitenrand)%
    \hbox{%
      \color{PraesentationTextfarbe}%
      \hbox{\insertframenavigationsymbol}%
      \hbox{\insertsubsectionnavigationsymbol}%
      \hbox{\insertsectionnavigationsymbol}%
    }%
  \end{textblock*}%
}

\newcommand{\PraesentationBildUhrenturm}{%
  \begin{textblock*}{0.426 \normPaperwidth}[1,1](\paperwidth - \PraesentationSeitenrand - \PraesentationSeitenrand, \paperheight - 0.063 \normPaperheight)%
    \includegraphics[width=0.426 \normPaperwidth]{ressources/TUM_Uhrenturm.png}%
  \end{textblock*}%
}

\newcommand{\PraesentationStartseiteUhrenturm}{%
  \setbeamertemplate{title page}{%
    \PraesentationSeitenkopfInhalt{ressources/Universitaet_Logo_RGB.pdf}%
    \PraesentationBildUhrenturm%
    \PraesentationTitelseiteInhalt%
  }%
}

\newcommand{\PraesentationStartseiteFlaggen}{%
  \setbeamertemplate{title page}{%
    \begin{textblock*}{\paperwidth}[0,1](-\PraesentationSeitenrand,\paperheight-\PraesentationSeitenrand)%
      \includegraphics[min width=\paperwidth,max width=\paperheight,min totalsize={\paperwidth}{\paperheight},keepaspectratio,center]{ressources/Universitaet_Flaggen.jpg}%
    \end{textblock*}%
    \PraesentationSeitenkopfInhalt{ressources/Universitaet_Logo_weiss.pdf}%
    \PraesentationTitelseiteInhalt%
  }%
}

\newcommand{\PraesentationStartseiteLeer}{%
  \setbeamertemplate{title page}{%
    \PraesentationSeitenkopfInhalt{ressources/Universitaet_Logo_weiss.pdf}%
    \PraesentationTitelseiteInhalt%
  }%
}


\newcommand{\PraesentationMasterStandard}{%
  \PraesentationFarbschemaStandard%
  \PraesentationStartseiteUhrenturm%
  \setbeamertemplate{headline}{%
    \PraesentationSeitenkopfInhalt{ressources/Universitaet_Logo_RGB.pdf}%
  }%
}

\newcommand{\PraesentationTitelseite}{\frame[plain]{\titlepage}}

\setbeamersize{
  text margin left=\PraesentationSeitenrand,
  text margin right=\PraesentationSeitenrand
}

% \setbeamertemplate{frametitle}{%
%   {\rule{0pt}{42mm + \PraesentationPositionKorrekturOben}\Large\selectfont\insertframetitle\newline\vspace*{-6.7mm}}%
% }
\setbeamertemplate{frametitle}{%
  \ifuseNiceTitles
  {\vspace*{0.04 \normPaperheight}\fontsize{14}{18}\selectfont \textrm \insertframetitle}%
  \else
  {\vspace*{0.04 \normPaperheight}\fontsize{14}{18}\selectfont\insertframetitle}%
  \fi
}

\setbeamertemplate{itemize items}[circle]
\setbeamertemplate{itemize subitem}[triangle]
\setbeamercolor{itemize subitem}{fg=black}
\setbeamerfont{itemize/enumerate subbody}{size=\normalsize}
% \addtolength{\leftmarginii}{16mm-2pt}%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DOKUMENT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% PDF-Einstellungen:
\hypersetup{
  pdfstartview={Fit},
  pdfproducer={\AllgemeinErsteller},
  pdfcreator={\AllgemeinGestalter}
}

\textblockorigin{\PraesentationSeitenrand}{\PraesentationSeitenrand} % Ursprung für Positionierung

\setbeamerfont{footnote}{size=\footnotesize}

\setbeamertemplate{footline}{
  \hbox{%
    \usebeamerfont{footnote}%
    \begin{beamercolorbox}{}%
      
      \hspace*{\PraesentationSeitenrand}%
      \ifuseNiceTitles
      \textrm{\PersonTitel{} \PersonVorname~\PersonNachname~(\UniversitaetAbkuerzung) \PraesentationFusszeileZusatz{}}%
      \hfill%
      \textrm{\insertframenumber{}}%
      \else
      \PersonTitel{} \PersonVorname~\PersonNachname~(\UniversitaetAbkuerzung) \PraesentationFusszeileZusatz{}%
      \hfill%
      \insertframenumber{}%
      \fi
      \hspace*{\PraesentationSeitenrand}%
    \end{beamercolorbox}%
    \vspace*{3.25mm}%
  }%
}

\setbeamertemplate{navigation symbols}{}



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Nice (square cornered) boxes
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% my packages
\RequirePackage[raster]{tcolorbox}

% The height is implemented a bit "hacky" now, should be improved
% "hacky" approach: title{\vphantom{Tg}#1}, this gives a fixed title height. Problem: the height of the title part is a bit too large; too much empty space above the text of the title.
% To make the title height change according to text height, use title={\strut #1} or adjusted title = #1.
\newtcolorbox{bluebox}[1]{width=\linewidth,colback={TUMBlau!5!white},title={\strut #1},arc=0mm, opacityframe=0, title filled, colbacktitle = TUMBlau, fonttitle=\large, toptitle=0mm, bottomtitle = 0mm, boxrule = 0pt}
\newtcolorbox{darkbluebox}[1]{width=\linewidth,colback={TUMBlau!20!white},title={\strut #1},arc=0mm, opacityframe=0, title filled, colbacktitle = TUMBlauDunkel, fonttitle=\large, toptitle=0mm, bottomtitle = 0mm, boxrule = 0pt}
\newtcolorbox{orangebox}[1]{width=\linewidth,colback={TUMOrange!12!white},title={\strut #1},arc=0mm, opacityframe=0, title filled, colbacktitle = TUMOrange, fonttitle=\large, toptitle=0mm, bottomtitle = 0mm, boxrule = 0pt}
\newtcolorbox{greenbox}[1]{width=\linewidth,colback={TUMGruenDunkel!12!white},title={\strut #1},arc=0mm, opacityframe=0, title filled, colbacktitle = TUMGruenDunkel, fonttitle=\large, toptitle=0mm, bottomtitle = 0mm, boxrule = 0pt}

% Environment Boxes
\newtcolorbox{thmbox}[1]{width=\linewidth,colback={TUMBlau!5!white},title={\strut Theorem (#1)},arc=0mm, opacityframe=0, title filled, colbacktitle = TUMBlau, fonttitle=\large, toptitle=0mm, bottomtitle = 0mm, boxrule = 0pt}
\newtcolorbox{defbox}[1]{width=\linewidth,colback={TUMOrange!12!white},title={\strut Definition (#1)},arc=0mm, opacityframe=0, title filled, colbacktitle = TUMOrange, fonttitle=\large, toptitle=0mm, bottomtitle = 0mm, boxrule = 0pt}
\newtcolorbox{egbox}[1]{width=\linewidth,colback={TUMGruenDunkel!12!white},title={\strut Example (#1)},arc=0mm, opacityframe=0, title filled, colbacktitle = TUMGruenDunkel, fonttitle=\large, toptitle=0mm, bottomtitle = 0mm, boxrule = 0pt}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Handout and notes
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Support for handout with notes

\if@notes

% Layouts handled by handoutWithNotes Package from
% https://github.com/gdiepen/latexbeamer-handoutWithNotes
\if@ruled
\RequirePackage[lines=12]{handoutWithNotes}
\else
\RequirePackage[lines=0]{handoutWithNotes}
\fi

\mode<handout>{\pgfpagesuselayout{1 on 1 with notes}[a4paper,border shrink=5mm]}
\fi
