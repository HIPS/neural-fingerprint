%\def\currversion{April 15, 2008}
\def\currversion{October 04, 2013}

  %*****************************************************************%
  %*                                                               *%
  %*     Prepared by Amy Hendrickson, TeXnology Inc.               *%
  %*     Copyright (C) 2004 by Amy Hendrickson, TeXnology Inc.,    *%
  %*     All rights reserved.                                      *%
  %*                                                               *%
  %*     TeXnology Inc.: 617 738-8029                              *%
  %*                     amyh@texnology.com                        *%
  %*                     www.texnology.com                         *%
  %*                                                               *%
  %*     For assistance, please contact the PNAS Editorial Office: *%
  %*     202-334-2679                                              *%
  %*     pnas@nas.edu                                              *%
  %*                                                               *%
  %*****************************************************************%

\typeout{^^J^^J
Two Column Journal Style for the^^J
Proceedings of the National Academy of Science
^^J
Written by Amy Hendrickson, TeXnology, Inc., 9/04^^J
^^J
This version is dated \currversion^^J^^J}

\ProvidesClass{pnastwo}
\newcommand\@ptsize{}
\newif\if@restonecol
\newif\if@titlepage
\@titlepagefalse
\if@compatibility\else
\DeclareOption{a4paper}
   {\setlength\paperheight {297mm}%
    \setlength\paperwidth  {210mm}}
\DeclareOption{a5paper}
   {\setlength\paperheight {210mm}%
    \setlength\paperwidth  {148mm}}
\DeclareOption{b5paper}
   {\setlength\paperheight {250mm}%
    \setlength\paperwidth  {176mm}}
\DeclareOption{letterpaper}
   {\setlength\paperheight {16in}%
    \setlength\paperwidth  {13.5in}}
\DeclareOption{legalpaper}
   {\setlength\paperheight {14in}%
    \setlength\paperwidth  {8.5in}}
\DeclareOption{executivepaper}
   {\setlength\paperheight {10.5in}%
    \setlength\paperwidth  {7.25in}}
\DeclareOption{landscape}
   {\setlength\@tempdima   {\paperheight}%
    \setlength\paperheight {\paperwidth}%
    \setlength\paperwidth  {\@tempdima}}
\DeclareOption{pnaspaper}
   {\setlength\paperheight {11.87in}%
    \setlength\paperwidth  {9.12in}}

\fi
\if@compatibility
  \renewcommand\@ptsize{0}
\else
\DeclareOption{10pt}{\renewcommand\@ptsize{0}}
\fi
\DeclareOption{11pt}{\renewcommand\@ptsize{1}}
\DeclareOption{12pt}{\renewcommand\@ptsize{2}}
\if@compatibility\else
\DeclareOption{oneside}{\@twosidefalse \@mparswitchfalse}
\fi
\DeclareOption{twoside}{\@twosidetrue  \@mparswitchtrue}
\DeclareOption{draft}{\setlength\overfullrule{5pt}}
\if@compatibility\else
\DeclareOption{final}{\setlength\overfullrule{0pt}}
\fi
\DeclareOption{titlepage}{\@titlepagetrue}
\if@compatibility\else
\DeclareOption{notitlepage}{\@titlepagefalse}
\fi
\if@compatibility\else
\DeclareOption{onecolumn}{\@twocolumnfalse}
\fi
\DeclareOption{twocolumn}{\@twocolumntrue}
\DeclareOption{leqno}{\input{leqno.clo}}
\DeclareOption{fleqn}{\input{fleqn.clo}}
\DeclareOption{openbib}{%
  \AtEndOfPackage{%
   \renewcommand\@openbib@code{%
      \advance\leftmargin\bibindent
      \itemindent -\bibindent
      \listparindent \itemindent
      \parsep \z@
      }%
   \renewcommand\newblock{\par}}%
}
\ExecuteOptions{pnaspaper,twoside,onecolumn,final}
\ProcessOptions

\AtBeginDocument{
\RequirePackage[cam,center]{crop}
\paperwidth=8.125in
\paperheight=10.875in
\renewcommand*\CROP@horigin{-.5truein}
\renewcommand*\CROP@vorigin{-.5truein}
}%


%% size10.clo ==>>
\renewcommand\normalsize{%
   \@setfontsize\normalsize\@ixpt\@xpt
   \abovedisplayskip 8\p@ \@plus2\p@ \@minus5\p@
   \abovedisplayshortskip \z@ \@plus3\p@
   \belowdisplayshortskip 6\p@ \@plus3\p@ \@minus3\p@
   \belowdisplayskip \abovedisplayskip
               \topsep 0\p@
               \parsep 0\p@
   \let\@listi\@listI}
\normalsize
               \topsep 0\p@
               \parsep 0\p@

\newcommand\small{%
   \@setfontsize\small\@viiipt{9}%
   \abovedisplayskip 8.5\p@ \@plus3\p@ \@minus4\p@
   \abovedisplayshortskip \z@ \@plus2\p@
   \belowdisplayshortskip 4\p@ \@plus2\p@ \@minus2\p@
   \def\@listi{\leftmargin\leftmargini
               \topsep 0\p@
               \parsep 0\p@
               \itemsep \parsep}%
   \belowdisplayskip \abovedisplayskip
}
\newcommand\footnotesize{%
   \@setfontsize\footnotesize\@vipt{7}%
   \abovedisplayskip 6\p@ \@plus2\p@ \@minus4\p@
   \abovedisplayshortskip \z@ \@plus\p@
   \belowdisplayshortskip 3\p@ \@plus\p@ \@minus2\p@
   \def\@listi{\leftmargin\leftmargini
               \topsep 3\p@ \@plus\p@ \@minus\p@
               \parsep 2\p@ \@plus\p@ \@minus\p@
               \itemsep \parsep}%
   \belowdisplayskip \abovedisplayskip
}
\newcommand\scriptsize{\@setfontsize\scriptsize\@vpt\@viipt}
\newcommand\tiny{\@setfontsize\tiny\@vpt\@vipt}
\newcommand\large{\@setfontsize\large\@xiipt{14}}
\newcommand\Large{\@setfontsize\Large\@xivpt{18}}
\newcommand\LARGE{\@setfontsize\LARGE\@xviipt{22}}
\newcommand\huge{\@setfontsize\huge\@xxpt{25}}
\newcommand\Huge{\@setfontsize\Huge\@xxvpt{30}}
\if@twocolumn
  \setlength\parindent{1em}
\else
  \setlength\parindent{9\p@}%15\p@}
\fi
\setlength\smallskipamount{3\p@ \@plus 1\p@ \@minus 1\p@}
\setlength\medskipamount{6\p@ \@plus 2\p@ \@minus 2\p@}
\setlength\bigskipamount{12\p@ \@plus 4\p@ \@minus 4\p@}
\setlength\headheight{12\p@}
\setlength\headsep   {25\p@}
\setlength\topskip   {10\p@}
\setlength\footskip{30\p@}
\if@compatibility \setlength\maxdepth{4\p@} \else
\setlength\maxdepth{.5\topskip} \fi
\if@compatibility
  \if@twocolumn
    \setlength\textwidth{410\p@}
  \else
    \setlength\textwidth{345\p@}
  \fi
\else
  \setlength\@tempdima{\paperwidth}
  \addtolength\@tempdima{-2in}
  \setlength\@tempdimb{345\p@}
  \if@twocolumn
    \ifdim\@tempdima>2\@tempdimb\relax
      \setlength\textwidth{2\@tempdimb}
    \else
      \setlength\textwidth{\@tempdima}
    \fi
  \else
    \ifdim\@tempdima>\@tempdimb\relax
      \setlength\textwidth{\@tempdimb}
    \else
      \setlength\textwidth{\@tempdima}
    \fi
  \fi
\fi
\if@compatibility\else
  \@settopoint\textwidth
\fi
\if@compatibility
  \setlength\textheight{43\baselineskip}
\else
  \setlength\@tempdima{\paperheight}
  \addtolength\@tempdima{-2in}
  \addtolength\@tempdima{-1.5in}
  \divide\@tempdima\baselineskip
  \@tempcnta=\@tempdima
  \setlength\textheight{\@tempcnta\baselineskip}
\fi
\addtolength\textheight{\topskip}
\if@twocolumn
 \setlength\marginparsep {10\p@}
\else
  \setlength\marginparsep{11\p@}
\fi
\setlength\marginparpush{5\p@}
\if@compatibility
  \if@twoside
     \setlength\oddsidemargin   {44\p@}
     \setlength\evensidemargin  {82\p@}
     \setlength\marginparwidth {107\p@}
  \else
     \setlength\oddsidemargin   {63\p@}
     \setlength\evensidemargin  {63\p@}
     \setlength\marginparwidth  {90\p@}
  \fi
  \if@twocolumn
     \setlength\oddsidemargin  {30\p@}
     \setlength\evensidemargin {30\p@}
     \setlength\marginparwidth {48\p@}
  \fi
\else
  \if@twoside
    \setlength\@tempdima        {\paperwidth}
    \addtolength\@tempdima      {-\textwidth}
    \setlength\oddsidemargin    {.4\@tempdima}
    \addtolength\oddsidemargin  {-1in}
    \setlength\marginparwidth   {.6\@tempdima}
    \addtolength\marginparwidth {-\marginparsep}
    \addtolength\marginparwidth {-0.4in}
  \else
    \setlength\@tempdima        {\paperwidth}
    \addtolength\@tempdima      {-\textwidth}
    \setlength\oddsidemargin    {.5\@tempdima}
    \addtolength\oddsidemargin  {-1in}
    \setlength\marginparwidth   {.5\@tempdima}
    \addtolength\marginparwidth {-\marginparsep}
    \addtolength\marginparwidth {-0.4in}
    \addtolength\marginparwidth {-.4in}
  \fi
  \ifdim \marginparwidth >2in
     \setlength\marginparwidth{2in}
  \fi
  \@settopoint\oddsidemargin
  \@settopoint\marginparwidth
  \setlength\evensidemargin  {\paperwidth}
  \addtolength\evensidemargin{-2in}
  \addtolength\evensidemargin{-\textwidth}
  \addtolength\evensidemargin{-\oddsidemargin}
  \@settopoint\evensidemargin
\fi
\if@compatibility
  \setlength\topmargin{27pt}
\else
  \setlength\topmargin{\paperheight}
  \addtolength\topmargin{-3.3in}
  \addtolength\topmargin{-\headheight}
  \addtolength\topmargin{-\headsep}
  \addtolength\topmargin{-\textheight}
  \addtolength\topmargin{-\footskip}     % this might be wrong!
  \addtolength\topmargin{-.5\topmargin}
  \@settopoint\topmargin
\fi
\setlength\footnotesep{6.65\p@}
\setlength{\skip\footins}{19\p@ \@plus 4\p@ \@minus 2\p@}
\setlength\floatsep    {12\p@ \@plus 2\p@ \@minus 2\p@}
\setlength\textfloatsep{20\p@ \@plus 2\p@ \@minus 4\p@}
\setlength\intextsep   {12\p@ \@plus 2\p@ \@minus 2\p@}
\setlength\dblfloatsep    {12\p@ \@plus 2\p@ \@minus 2\p@}
\setlength\dbltextfloatsep{20\p@ \@plus 2\p@ \@minus 4\p@}
\setlength\@fptop{0\p@ \@plus 1fil}
\setlength\@fpsep{8\p@ \@plus 2fil}
\setlength\@fpbot{0\p@ \@plus 1fil}
\setlength\@dblfptop{0\p@ \@plus 1fil}
\setlength\@dblfpsep{8\p@ \@plus 2fil}
\setlength\@dblfpbot{0\p@ \@plus 1fil}
\setlength\partopsep{2\p@ \@plus 1\p@ \@minus 1\p@}
\def\@listi{\leftmargin\leftmargini
            \parsep 1sp%4\p@ \@plus2\p@ \@minus\p@
            \topsep 8\p@ \@plus2\p@ \@minus4\p@
            \itemsep1sp%4\p@ \@plus2\p@ \@minus\p@
}
\let\@listI\@listi
\@listi

\def\@listii {\leftmargin\leftmarginii
              \labelwidth\leftmarginii
              \advance\labelwidth-\labelsep
              \topsep    4\p@ \@plus2\p@ \@minus\p@
              \parsep    2\p@ \@plus\p@  \@minus\p@
              \itemsep   \parsep}
\def\@listiii{\leftmargin\leftmarginiii
              \labelwidth\leftmarginiii
              \advance\labelwidth-\labelsep
              \topsep    2\p@ \@plus\p@\@minus\p@
              \parsep    \z@
              \partopsep \p@ \@plus\z@ \@minus\p@
              \itemsep   \topsep}
\def\@listiv {\leftmargin\leftmarginiv
              \labelwidth\leftmarginiv
              \advance\labelwidth-\labelsep}
\def\@listv  {\leftmargin\leftmarginv
              \labelwidth\leftmarginv
              \advance\labelwidth-\labelsep}
\def\@listvi {\leftmargin\leftmarginvi
              \labelwidth\leftmarginvi
              \advance\labelwidth-\labelsep}


%% <<== size10.clo

\setlength\lineskip{1\p@}
\setlength\normallineskip{1\p@}
\renewcommand\baselinestretch{}
\setlength\parskip{0\p@ \@plus \p@}
\@lowpenalty   51
\@medpenalty  151
\@highpenalty 301
\setcounter{topnumber}{2}
\renewcommand\topfraction{.7}
\setcounter{bottomnumber}{1}
\renewcommand\bottomfraction{.3}
\setcounter{totalnumber}{3}
\renewcommand\textfraction{.2}
\renewcommand\floatpagefraction{.5}
\setcounter{dbltopnumber}{2}
\renewcommand\dbltopfraction{.7}
\renewcommand\dblfloatpagefraction{.5}

\newif\iftitlepage
  \def\ps@headings{%
      \let\@evenhead\relax
   \def\@oddhead{\hfill\rlap{\hskip14pt\vtop to 0pt{\vskip-1pt\makesidesquares
\vss}}}%
\def\@oddfoot{%
%\leftnumber
{\dospecialaccents%
\iftitlepage\global\titlepagefalse%
\footlinefont
\ifodd\c@page
\hbox to \textwidth{\footlinefont \let\it\footlineit
\theurl\hfill\footlinefont PNAS\hskip6pt  \vrule depth 2pt\hskip6pt
{\foliofont\theissuedate}\hskip6pt\vrule\hskip6pt\thevolume\hskip6pt
\vrule\hskip6pt\theissuenumber\hskip6pt \vrule\hskip6pt
{\foliofont\the\c@page{\footlineit--}\expandafter\ifx\csname
lastpage\endcsname\relax ??\else\lastpage\fi}}%
\else
\hbox to \textwidth{\footlinefont \let\it\footlineit
{\foliofont\the\c@page{\footlineit--}\expandafter\ifx\csname
lastpage\endcsname\relax ??\else\lastpage\fi}\hskip6pt
\vrule depth 2pt\hskip6pt PNAS\hskip6pt
\vrule\hskip6pt{\foliofont\theissuedate}\hskip6pt\vrule\hskip6pt
\thevolume\hskip6pt\vrule\hskip6pt\theissuenumber\hfill
\theurl}%
\fi\else%
\ifodd\c@page
\hbox to\textwidth{{\let\it\footlineit\footlinefont\thefootlineauthor}\hfill
\footlinefont PNAS\hskip6pt\vrule depth 2pt\hskip6pt{\foliofont\theissuedate}%
\hskip6pt\vrule depth2pt\hskip6pt\thevolume\hskip6pt\vrule depth 2pt\hskip6pt\theissuenumber\hskip6pt\vrule\hskip6pt{\foliofont\the\c@page}}%
\else%
\hbox to\textwidth{\let\it\footlineit\footlinefont{\foliofont \the\c@page}%
\hskip6pt\vrule depth 2pt height 6pt\hskip6pt\theurl\hfill\thefootlineauthor}\fi\fi}%\rightnumber
}
\let\@evenfoot\@oddfoot
%
      \let\@mkboth\markboth
    \def\sectionmark##1{%
      \markboth {\MakeUppercase{%
        \ifnum \c@secnumdepth >\z@
          \thesection\quad
        \fi
        ##1}}{}}%
    \def\subsectionmark##1{%
      \markright {%
        \ifnum \c@secnumdepth >\@ne
          \thesubsection\quad
        \fi
        ##1}}}


\newcount\affilcount
\newcount\saveaffilcount
\newcount\c@fnaffilcount
\def\affil#1#2{\normalsize\affilfont\ \affilcount=#1
\ifnum\affilcount>\saveaffilcount\global\saveaffilcount\affilcount\fi
\expandafter\ifx\csname affil\the\affilcount\endcsname\relax
\global\advance\c@footnote by1
\expandafter\xdef\csname affilsymb\the\affilcount\endcsname%
{\the\c@footnote}{$^{\@fnsymbol\c@footnote}$}%
\expandafter\gdef\csname affil\the\affilcount\endcsname{#2}\else
$^{\expandafter\@fnsymbol\csname affilsymb\the\affilcount\endcsname}$
\fi\authorfont}

\newcommand\maketitle{\par
\global\titlepagetrue
  \begingroup
\dospecialaccents
\parindent=0pt
\def\and{ and }
    \renewcommand\thefootnote{\@fnsymbol\c@footnote}%
    \def\@makefnmark{\rlap{\@textsuperscript{\normalfont\@thefnmark}}}%
    \long\def\@makefntext##1{\parindent0pt\noindent
                \@textsuperscript{\normalfont\@thefnmark}##1}%
      \newpage
      \global\@topnum\z@   % Prevents figures from going at top of page.
      \@maketitle
\@thanks
% according to sample, Dec 2007, copyright no longer wanted here
%    \renewcommand\thefootnote{}%
% \footnotetext{\vrule height 8pt width0pt\copyright \thecopyrightyear\ by The % National Academy of Sciences of the USA}
  \endgroup
  \setcounter{footnote}{0}%
  \global\let\thanks\relax
  \global\let\maketitle\relax
  \global\let\@maketitle\relax
  \global\let\@thanks\@empty
  \global\let\@author\@empty
  \global\let\@date\@empty
  \global\let\@title\@empty
  \global\let\@contributor\@empty
  \global\let\title\relax
  \global\let\date\relax
  \global\let\and\relax
}
\def\@maketitle{%
\global\firsttimetrue
%% To get equation numbers in square brackets and bold
\gdef\tagform@##1{\maketag@@@{\bf[\hskip1pt\ignorespaces##1\unskip\@@italiccorr\hskip1pt]}}
\gdef\@eqnnum{{\normalfont\bf \normalcolor [\hskip1pt\theequation\hskip1pt]}}
  \newpage
\def\\ {\vskip1pt}
  \let \footnote \thanks
    {\hsize=\textwidth \raggedright
\noindent\huge\baselineskip= 24pt\titlefont \@title \par}%
    \vskip6pt
    {\normalsize\authorfont
        \@author
\par
{\vskip4pt
\raggedright
\affilcount=0
\footnotesize\affilfont
\loop \ifnum\affilcount< \saveaffilcount
\expandafter\ifx\csname affil\the\affilcount\endcsname\relax\else
$^{\expandafter\@fnsymbol\csname affilsymb\the\affilcount\endcsname}$%
\csname affil\the\affilcount\endcsname,\fi
\global\advance\affilcount by1
\repeat\ifnum\saveaffilcount=1\else\ and \fi
$^{\expandafter\@fnsymbol\csname affilsymb\the\affilcount\endcsname}$%
\expandafter\csname affil\the\affilcount\endcsname
}}%
    \vskip 4pt%
    {\affilfont \@contributor}%
  \par
  \vskip 1.5em}

\def\contributor#1{\def\@contributor{#1}}
\let\@contributor\relax

\contributor{Submitted to Proceedings
of the National Academy of Sciences of the United States of America}



\setcounter{secnumdepth}{3}
\newcounter {part}
\newcounter {section}
\newcounter {subsection}[section]
\newcounter {subsubsection}[subsection]
\newcounter {paragraph}[subsubsection]
\newcounter {subparagraph}[paragraph]
\renewcommand \thepart {\@Roman\c@part}
\renewcommand \thesection {\@arabic\c@section}
\renewcommand\thesubsection   {\thesection.\@arabic\c@subsection}
\renewcommand\thesubsubsection{\thesubsection .\@arabic\c@subsubsection}
\renewcommand\theparagraph    {\thesubsubsection.\@arabic\c@paragraph}
\renewcommand\thesubparagraph {\theparagraph.\@arabic\c@subparagraph}
\newcommand\part{%
   \if@noskipsec \leavevmode \fi
   \par
   \addvspace{4ex}%
   \@afterindentfalse
   \secdef\@part\@spart}

\def\@part[#1]#2{%
    \ifnum \c@secnumdepth >\m@ne
      \refstepcounter{part}%
      \addcontentsline{toc}{part}{\thepart\hspace{1em}#1}%
    \else
      \addcontentsline{toc}{part}{#1}%
    \fi
    {\parindent \z@ \raggedright
     \interlinepenalty \@M
     \normalfont
     \ifnum \c@secnumdepth >\m@ne
       \Large\bfseries \partname~\thepart
       \par\nobreak
     \fi
     \huge \bfseries #2%
     \markboth{}{}\par}%
    \nobreak
    \vskip 3ex
    \@afterheading}
\def\@spart#1{%
    {\parindent \z@ \raggedright
     \interlinepenalty \@M
     \normalfont
     \huge \bfseries #1\par}%
     \nobreak
     \vskip 3ex
     \@afterheading}

\def\@startsection#1#2#3#4#5#6{%
  \if@noskipsec \leavevmode \fi
  \par
  \@tempskipa #4\relax
  \@afterindenttrue
  \ifdim \@tempskipa <\z@
    \@tempskipa -\@tempskipa \@afterindentfalse
  \fi
  \if@nobreak
    \everypar{}%
  \else
    \addpenalty\@secpenalty\addvspace\@tempskipa
  \fi
  \@ifstar
    {\@ssect{#3}{#4}{#5}{#6}}%
    {\@dblarg{\@sect{#1}{#2}{#3}{#4}{#5}{#6}}}}

\def\@sect#1#2#3#4#5#6[#7]#8{%
  \ifnum #2>\c@secnumdepth
    \let\@svsec\@empty
  \else
    \refstepcounter{#1}%
    \protected@edef\@svsec{\@seccntformat{#1}\relax}%
  \fi
  \@tempskipa #5\relax
  \ifdim \@tempskipa>\z@
    \begingroup
      #6{%
        \@hangfrom{\hskip #3\relax\@svsec
}%
          \interlinepenalty \@M #8\@@par}%
    \endgroup
    \csname #1mark\endcsname{#7}%
    \addcontentsline{toc}{#1}{%
      \ifnum #2>\c@secnumdepth \else
        \protect\numberline{\csname the#1\endcsname}%
      \fi
      #7}%
  \else
    \def\@svsechd{%
      #6{\hskip #3\relax
      \@svsec #8\ifnum#2=2.\fi}% <<==
      \csname #1mark\endcsname{#7}%
      \addcontentsline{toc}{#1}{%
        \ifnum #2>\c@secnumdepth \else
          \protect\numberline{\csname the#1\endcsname}%
        \fi
        #7}}%
  \fi
  \@xsect{#5}}

\newcommand\section{\@startsection {section}{1}{\z@}%
                                   {-10pt \@plus -1ex \@minus -.2ex}%
                                   {.5ex }%
                                   {\normalfont\Large\bfseries\sectionfont}}
\newcommand\subsection{\@startsection{subsection}{2}{\z@}%
                                     {10pt\@plus 1ex \@minus .2ex}%
                                     {-0.5ex \@plus .2ex}%
                                     {\normalfont\large\bfseries\subsectionfont}}
\newcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
                                     {-3.25ex\@plus -1ex \@minus -.2ex}%
                                     {1.5ex \@plus .2ex}%
                                     {\normalfont\normalsize\bfseries}}
\newcommand\paragraph{\@startsection{paragraph}{4}{\z@}%
                                    {3.25ex \@plus1ex \@minus.2ex}%
                                    {-1em}%
                                    {\normalfont\normalsize\bfseries}}
\newcommand\subparagraph{\@startsection{subparagraph}{5}{\parindent}%
                                       {3.25ex \@plus1ex \@minus .2ex}%
                                       {-1em}%
                                      {\normalfont\normalsize\bfseries}}

\setlength\leftmargini {12pt}
\leftmargin  \leftmargini
\setlength\leftmarginii  {12pt}
\setlength\leftmarginiii {12pt}
\setlength\leftmarginiv  {12pt}

\if@twocolumn
  \setlength\leftmarginv  {.5em}
  \setlength\leftmarginvi {.5em}
\else
  \setlength\leftmarginv  {1em}
  \setlength\leftmarginvi {1em}
\fi
\setlength  \labelsep  {.5em}
\setlength  \labelwidth{\leftmargini}
\addtolength\labelwidth{-\labelsep}
\@beginparpenalty -\@lowpenalty
\@endparpenalty   -\@lowpenalty
\@itempenalty     -\@lowpenalty
\renewcommand\theenumi{\@arabic\c@enumi}
\renewcommand\theenumii{\@alph\c@enumii}
\renewcommand\theenumiii{\@roman\c@enumiii}
\renewcommand\theenumiv{\@Alph\c@enumiv}
\newcommand\labelenumi{\theenumi.}
\newcommand\labelenumii{(\theenumii)}
\newcommand\labelenumiii{\theenumiii.}
\newcommand\labelenumiv{\theenumiv.}
\renewcommand\p@enumii{\theenumi}
\renewcommand\p@enumiii{\theenumi(\theenumii)}
\renewcommand\p@enumiv{\p@enumiii\theenumiii}
\newcommand\labelitemi{\raise1.5pt\llap{\small\textbullet\hskip2pt}}
\newcommand\labelitemii{\normalfont\bfseries \textendash}
\newcommand\labelitemiii{\textasteriskcentered}
\newcommand\labelitemiv{\textperiodcentered}
\newenvironment{description}
               {\list{}{\labelwidth\z@ \itemindent-\leftmargin
                        \let\makelabel\descriptionlabel}}
               {\endlist}
\newcommand*\descriptionlabel[1]{\hspace\labelsep
                                \normalfont\bfseries #1}

\def\abstract{\bgroup
\parindent=0pt
\hsize=\columnwidth
\font\it\helveticaboldoblique at 8pt
\font\bf\helveticabold at 8pt
\def\textit##1{{\it ##1}}
\def\textbf##1{{\bf ##1}}
\small
\abstractfont\dospecialaccents}

\def\endabstract{\vskip11pt
\egroup}



\newenvironment{verse}
               {\let\\\@centercr
                \list{}{\itemsep      \z@
                        \itemindent   -1.5em%
                        \listparindent\itemindent
                        \rightmargin  \leftmargin
                        \advance\leftmargin 1.5em}%
                \item\relax}
               {\endlist}
\newenvironment{quotation}
               {\list{}{\listparindent 1.5em%
                        \itemindent    \listparindent
                        \rightmargin   \leftmargin
                        \parsep        \z@ \@plus\p@}%
                \item\relax}
               {\endlist}
\newenvironment{quote}
               {\list{}{\rightmargin\leftmargin}%
                \item\relax}
               {\endlist}
\if@compatibility
\newenvironment{titlepage}
    {%
      \if@twocolumn
        \@restonecoltrue\onecolumn
      \else
        \@restonecolfalse\newpage
      \fi
      \setcounter{page}\z@
    }%
    {\if@restonecol\twocolumn \else \newpage \fi
    }
\else
\newenvironment{titlepage}
    {%
      \if@twocolumn
        \@restonecoltrue\onecolumn
      \else
        \@restonecolfalse\newpage
      \fi
      \setcounter{page}\@ne
    }%
    {\if@restonecol\twocolumn \else \newpage \fi
     \if@twoside\else
        \setcounter{page}\@ne
     \fi
    }
\fi
\newcommand\appendix{\par
  \setcounter{section}{0}%
  \setcounter{subsection}{0}%
  \gdef\thesection{\@Alph\c@section}}
\setlength\arraycolsep{5\p@}
\setlength\tabcolsep{6\p@}
\setlength\arrayrulewidth{.4\p@}
\setlength\doublerulesep{2\p@}
\setlength\tabbingsep{\labelsep}
\skip\@mpfootins = \skip\footins
\setlength\fboxsep{3\p@}
\setlength\fboxrule{.4\p@}
\renewcommand \theequation {\@arabic\c@equation}
\newcounter{figure}
\renewcommand \thefigure {\@arabic\c@figure}
\def\fps@figure{tbp}
\def\ftype@figure{1}
\def\ext@figure{lof}
\def\fnum@figure{\figurename~\thefigure}
\newenvironment{figure}
               {\@float{figure}}
               {\end@float}
\newenvironment{figure*}
               {\@dblfloat{figure}}
               {\end@dblfloat}
\newcounter{table}
\renewcommand\thetable{\@arabic\c@table}
\def\fps@table{tbp}
\def\ftype@table{2}
\def\ext@table{lot}
\def\fnum@table{\tablename~\thetable}
\newenvironment{table}
               {\@float{table}}
               {\end@float}
\newenvironment{table*}
               {\@dblfloat{table}}
               {\end@dblfloat}


\newenvironment{xtable}
               {\@dblfloat{table}}
               {\end@dblfloat}
\newenvironment{xfigure}
               {\@dblfloat{figure}}
               {\end@dblfloat}

\newlength\abovecaptionskip
\newlength\belowcaptionskip
\setlength\abovecaptionskip{2\p@}
\setlength\belowcaptionskip{4\p@}

\long\def\@makecaption#1#2{%
\ifx\@captype\xtable
\let\currtabcaption\relax
%\gdef\currtabcaption{%
%\tabnumfont\relax #1.\tabtextfont\hskip8\p@\relax#2\par
%\vskip\belowcaptionskip
%}
\vskip\abovecaptionskip
  \sbox\@tempboxa{\tabnumfont#1.\tabtextfont\hskip8\p@\relax #2}%
  \ifdim \wd\@tempboxa >\hsize
\tabnumfont\relax #1.\tabtextfont\hskip8\p@\relax#2\par
  \else
    \global \@minipagefalse
    \hb@xt@\hsize{\hfil\box\@tempboxa\hfil}%
  \fi
\else
 \vskip\abovecaptionskip
  \sbox\@tempboxa{\fignumfont#1.\figtextfont\hskip8\p@\relax #2}%
  \ifdim \wd\@tempboxa >\hsize
\fignumfont\relax #1.\figtextfont\hskip8\p@\relax#2\par
  \else
    \global \@minipagefalse
    \hb@xt@\hsize{\hfil\box\@tempboxa\hfil}%
  \fi
\fi
}
\DeclareOldFontCommand{\rm}{\normalfont\rmfamily}{\mathrm}
\DeclareOldFontCommand{\sf}{\normalfont\sffamily}{\mathsf}
\DeclareOldFontCommand{\tt}{\normalfont\ttfamily}{\mathtt}
\DeclareOldFontCommand{\bf}{\normalfont\bfseries}{\mathbf}
\DeclareOldFontCommand{\it}{\normalfont\itshape}{\mathit}
\DeclareOldFontCommand{\sl}{\normalfont\slshape}{\@nomath\sl}
\DeclareOldFontCommand{\sc}{\normalfont\scshape}{\@nomath\sc}
\DeclareRobustCommand*\cal{\@fontswitch\relax\mathcal}
\DeclareRobustCommand*\mit{\@fontswitch\relax\mathnormal}
\newcommand\@pnumwidth{1.55em}
\newcommand\@tocrmarg{2.55em}
\newcommand\@dotsep{4.5}
\setcounter{tocdepth}{3}
\newcommand\tableofcontents{%
    \section*{\contentsname
        \@mkboth{%
           \MakeUppercase\contentsname}{\MakeUppercase\contentsname}}%
    \@starttoc{toc}%
    }
\newcommand*\l@part[2]{%
  \ifnum \c@tocdepth >-2\relax
    \addpenalty\@secpenalty
    \addvspace{2.25em \@plus\p@}%
    \begingroup
      \parindent \z@ \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      {\leavevmode
       \large \bfseries #1\hfil \hb@xt@\@pnumwidth{\hss #2}}\par
       \nobreak
       \if@compatibility
         \global\@nobreaktrue
         \everypar{\global\@nobreakfalse\everypar{}}%
      \fi
    \endgroup
  \fi}
\newcommand*\l@section[2]{%
  \ifnum \c@tocdepth >\z@
    \addpenalty\@secpenalty
    \addvspace{1.0em \@plus\p@}%
    \setlength\@tempdima{1.5em}%
    \begingroup
      \parindent \z@ \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      \leavevmode \bfseries
      \advance\leftskip\@tempdima
      \hskip -\leftskip
      #1\nobreak\hfil \nobreak\hb@xt@\@pnumwidth{\hss #2}\par
    \endgroup
  \fi}
\newcommand*\l@subsection{\@dottedtocline{2}{1.5em}{2.3em}}
\newcommand*\l@subsubsection{\@dottedtocline{3}{3.8em}{3.2em}}
\newcommand*\l@paragraph{\@dottedtocline{4}{7.0em}{4.1em}}
\newcommand*\l@subparagraph{\@dottedtocline{5}{10em}{5em}}
\newcommand\listoffigures{%
    \section*{\listfigurename
      \@mkboth{\MakeUppercase\listfigurename}%
              {\MakeUppercase\listfigurename}}%
    \@starttoc{lof}%
    }
\newcommand*\l@figure{\@dottedtocline{1}{1.5em}{2.3em}}
\newcommand\listoftables{%
    \section*{\listtablename
      \@mkboth{%
          \MakeUppercase\listtablename}{\MakeUppercase\listtablename}}%
    \@starttoc{lot}%
    }
\let\l@table\l@figure
\newdimen\bibindent
\setlength\bibindent{1.5em}

\newskip\bibskip
\bibskip=0pt plus 3pt
\newenvironment{thebibliography}[1]
     {\endgroup\endtwocolumns\vskip24pt\twocolumns\begingroup
\def\@currenvir{thebibliography}
\let\savebibitem\bibitem
\def\bibitem{\vskip\bibskip\savebibitem}
\tiny\baselineskip=8pt\bibfont
\let\it\bibitfont
\let\bf\bibbffont
\let\em\bibitfont
\let\sc\bibfont
\def\textit##1{{\bibitfont ##1}}
\def\textbf##1{{\bibbffont ##1}}
      \list{\@biblabel{\@arabic\c@enumiv}}%
           {\settowidth\labelwidth{\@biblabel{#1}}%
            \leftmargin\labelwidth
            \advance\leftmargin\labelsep
            \@openbib@code
            \usecounter{enumiv}%
            \let\p@enumiv\@empty
            \renewcommand\theenumiv{\@arabic\c@enumiv}}%
      \sloppy
      \clubpenalty4000
      \@clubpenalty \clubpenalty
      \widowpenalty4000%
      \sfcode`\.\@m}
     {\def\@noitemerr
       {\@latex@warning{Empty `thebibliography' environment}}%
      \endlist
}
\newcommand\newblock{\hskip .11em\@plus.33em\@minus.07em}
\let\@openbib@code\@empty

%\renewcommand\footnoterule{%
%  \kern-3\p@
%  \hrule\@width6pc%.4\columnwidth
%  \kern2.6\p@}
\renewcommand\footnoterule{%
\ifx \@significancetext\@empty
			\kern-3\p@
				\hrule\@width6pc%.4\columnwidth
				\kern2.6\p@
\else
			\ifnum\thepage=1
				\renewcommand\footnoterule{}
			\else
				\kern-3\p@
				\hrule\@width6pc%.4\columnwidth
				\kern2.6\p@
			\fi
\fi
			}
\newcommand\@makefntext[1]{%
    \parindent 1em%
    \noindent
    \hb@xt@1.8em{\hss\@makefnmark}#1}
\newcommand\contentsname{Contents}
\newcommand\listfigurename{List of Figures}
\newcommand\listtablename{List of Tables}
\newcommand\refname{References}
\newcommand\indexname{Index}
\newcommand\figurename{Fig.}
\newcommand\tablename{Table}
\newcommand\partname{Part}
\newcommand\appendixname{Appendix}
\newcommand\abstractname{Abstract}
\def\today{\ifcase\month\or
  January\or February\or March\or April\or May\or June\or
  July\or August\or September\or October\or November\or December\fi
  \space\number\day, \number\year}
\setlength\columnsep{10\p@}
\setlength\columnseprule{0\p@}
\pagenumbering{arabic}
\if@twoside
\else
  \raggedbottom
\fi
\if@twocolumn
  \twocolumn
  \sloppy
  \flushbottom
\else
  \onecolumn
\fi

%%%%%%%%%%%%%%%%%
% Additions to article.cls for PNASONE ==>>
%%%%%%%%%%%%%%%%%
\hoffset-.5in
\pagestyle{headings}
\textwidth= 42.5pc
\textheight= 56.2pc

\def\xrelax{\relax}



%%%%%%%%%%%%%%%%%%%%%%%% For Line numbers %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newcounter{cms}

\def\leftnumber{\setlength{\unitlength}{1pt}
\begin{picture}(0,0)
\multiput(-45,688)(0,-10){68}{\addtocounter
{cms}{1}{\fontsize{9}{10.5}\selectfont\arabic{cms}\par}}
\end{picture}
}

\def\rightnumber{\setlength{\unitlength}{1pt}
\begin{picture}(0,0)
\multiput(6,688)(0,-10){68}{\addtocounter
{cms}{1}{\fontsize{9}{10.5}\selectfont\arabic{cms}\par}}
\end{picture}
}
%%%%%%%%%%%%%%%%%%%%%%%% For Line numbers %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%
%% Footnotes, endnotes

\newcount\footnum
\newcount\savefootnum

\def\notes{
{\ifnum\footnum=0\else
\savefootnum=\footnum
\footnum=0
\noindent\loop\ifnum\savefootnum>\footnum%
\global\advance\footnum by1\relax%
\csname foot\the\footnum\endcsname\relax%
\expandafter\gdef\csname foot\the\footnum\endcsname{\relax}\relax%
\repeat\global\footnum=0\relax\fi}}


\long\def\endnotes#1{\global\advance\footnum by 1\relax$^{\the\footnum}$%
\long\expandafter\gdef\csname foot\the\footnum\endcsname{%
\vbox{\footnotesize\everypar={\hskip-10pt\everypar={}}
\leftskip=10pt\relax
\noindent\hbox to10pt{\the\footnum.\hfill}{\def\@currentlabel{\the\footnum}%
#1\strut\vskip1sp}}\vskip1pt}}

\def\endnotetext#1{\global\advance\footnum by 1\relax
\long\expandafter\gdef\csname foot\the\footnum\endcsname{%
\vtop{\footnotesize%
\leftskip=12pt\relax\parindent=-12pt
\indent\hbox to12pt{\the\footnum.\hfill}{#1\strut\vskip1sp}}\vskip1pt}}

%\let\footnote\endnotes

     \setlength\oddsidemargin   {3\p@}
     \setlength\evensidemargin   {-3\p@}


\def\dropcap#1{\setbox1=\hbox{\dropcapfont\uppercase{#1}\hskip1pt}
\hangindent=\wd1
\hangafter-2
\noindent\llap{\vbox to0pt{\vskip-7pt\copy1\vss}}}

%%% Theorem environment %%%%%%%%%

%% added \theoremfont for theorem name and number

\def\@begintheorem#1#2{\trivlist
   \item[\hskip \labelsep{\theoremfont #1\ #2.}]\itshape}
\def\@opargbegintheorem#1#2#3{\trivlist
      \item[\hskip \labelsep{\theoremfont #1\ #2.\ (#3)}]\itshape}
\def\@endtheorem{\endtrivlist}

%%%%%%
%%Macros for Footline on titlepage

\def\url#1{\def\theurl{#1}}
\def\issuedate#1{\def\theissuedate{#1}}
\def\copyrightyear#1{\def\thecopyrightyear{#1}}
\def\volume#1{\def\thevolume{#1}}
\def\issuenumber#1{\def\theissuenumber{#1}}
\def\footlineauthor#1{\def\thefootlineauthor{\let\it\footlineit#1}}

\ps@headings

%% References

\def\@biblabel#1{#1.}

%%% code to get text in tables to extend all the way to left and right
%%% of table

\def\xtable{table}
\def\@array[#1]#2{\tabletextfont
\setbox\@arstrutbox=\hbox{\vrule
     height\arraystretch \ht\strutbox
     depth\arraystretch \dp\strutbox
     width\z@}\@mkpream{#2}\edef\@preamble{\halign \noexpand\@halignto
\bgroup%
\tabskip\z@\@arstrut\@preamble
\ifx\@captype\xtable\hskip-\tabcolsep\fi%% <==== Changed
\tabskip\z@ \cr}%
\let\@startpbox\@@startpbox \let\@endpbox\@@endpbox%
  \if #1t\vtop \else \if#1b\vbox \else \vcenter \fi\fi%
  \bgroup\let\par\relax%
  \let\@sharp##\let\protect\relax \lineskip\z@\baselineskip\z@\@preamble}

%% new
\def\new@tabacol{\edef\@preamble{\@preamble\hskip0pt}}

\def\@tabclassz{\ifcase \@lastchclass\@acolampacol%
\or \@ampacol \or
   \or \or \@addamp \or \@acolampacol\or \@firstampfalse
\ifx\@captype\xtable \new@tabacol\else\@tabacol \fi%
\fi%
\edef\@preamble{\@preamble%
  \ifcase \@chnum%
     \hfil\ignorespaces\@sharp\unskip\hfil%
     \or \ignorespaces\@sharp\unskip\hfil%
     \or \hfil\hskip\z@ \ignorespaces\@sharp\unskip\fi}}

\setcounter{secnumdepth}{0}

    \long\def\@makefntext#1{\parindent0pt\noindent
                \@textsuperscript{\normalfont\@thefnmark}#1}%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                                                                      %%
%% LaTeX Style for Multiple Columns with Floating Insertions            %%
%%                                                                      %%
%% Copyright 2004, Amy Hendrickson, TeXnology Inc.                      %%
%% All rights reserved                                                  %%
%%                                                                      %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%
%%% Double column output routines,
%%% with single and double column insertions,
%%% to be positioned at page or column top, page or column bottom, page
%%% or column center, or full page floating insertion.
%%% This code was originally based on Frank Mittlebach's multicol.sty.
%%% The concept of how to balance columns is due to Donald Knuth. The
%%% way to integrate the changed output routine into the standard LaTeX
%%% output routine is due to Frank.
%%% The method of making the single and double column insertions and the
%%% changes in the output routines to accomodate these features are mine.
%%% A.H., August 2004
%%

\newskip\abovefigcaptionskip
\newskip\abovetabcaptionskip
\newskip\abovetableskip
\newif\ifcentermultiplelines
\newif\ifcentersingleline
\newif\ifcentersingletabline
\newif\iflineabovetabcaption
\newif\iflinebelowtabcaption
\newdimen\belowtabcaptionskip

%% Parameters you can set:
\columnsep=13.5pt%16pt
\newdimen\saveparskip
\saveparskip=\parskip

%%
\newdimen\captionskip %% space between caption and figure or table
\captionskip=0pt
%%%

\newdimen\PushOneColTopFig
\newdimen\PushOneColBotFig

\PushOneColTopFig=5.5pt %% push down from top of text page
\PushOneColBotFig=-4.5pt %

%%
\newdimen\PushOneColTopTab
\newdimen\PushOneColBotTab

\PushOneColTopTab=2.5pt  %% push down from top of text page
\PushOneColBotTab=0pt %% push up from bottom of text page

%%
\newdimen\PushTwoColTopFig
\newdimen\PushTwoColBotFig

\PushTwoColTopFig=2.5pt %% push down from top of text page
\PushTwoColBotFig=1.5pt   %% push up from bottom of text page

%%
\newdimen\PushTwoColTopTab
\newdimen\PushTwoColBotTab

\PushTwoColTopTab=1.5pt %% push down from top of text page
\PushTwoColBotTab=1.5pt %% push up from bottom of text page

%%

\newskip\intextfloatskip
\intextfloatskip=20pt %%  Space between figure or table and text.
         %% (Need to subtract 4pt to get what you want)

%% Normal and Appendix Figure and Table Numbers

\def\appthefigure{\Alph{section}\the\c@appfigure\theletter}
\def\appthetable{\Alph{section}\the\c@apptable\theletter}
\def\apptheplate{\Alph{section}\the\c@appplate\theletter}

\newcount\c@appfigure
\newcount\c@apptable
\newcount\c@appplate

%% <=== end of parameters you can set

%% do one column at a time so that we can get single col
%% floats for first col.

%%% New version of \@makecol that includes dbltopins and dblbotins,
%%% insertions that span both columns at the top and bottom of text page.

%%%

\def\new@makecol{\setbox\@outputbox
     \vbox{\boxmaxdepth \maxdepth
\ifdim\ht\dbltopins<1pt\else\unvbox\dbltopins\fi
     \unvbox\@cclv
\ifdim\ht\dblbotins<1pt\else\unvbox\dblbotins\fi%
\ifvoid\footins\else\vskip\skip\footins\footnoterule\unvbox\footins\fi\vss}
%\global\savefigandtabnumber\figandtabnumber
%\global\advance\savefigandtabnumber by 1 %% Because loop stops one short
                                         %% of the total number of figs
\global\savedblfigandtabnumber\dblfigandtabnumber
   \xdef\@freelist{\@freelist\@midlist}\gdef\@midlist{}\@combinefloats
   \setbox\@outputbox\vbox to\@colht{\boxmaxdepth\maxdepth
   \@texttop\dimen128=\dp\@outputbox\unvbox\@outputbox
   \vskip-\dimen128\@textbottom}%
   \global\maxdepth\@maxdepth}

%%
\let\@makecol\new@makecol %%

%% Default Latex output routine, to return to after balance columns
%% This should make it possible to capture the output routine of
%% any version of 2e as well as 2.09. ==>>

\edef\curroutput{\the\output}
\let\latexoutput\curroutput

%% table* and figure*  outside two columns keep default definitions.

\newif\ifnonewpage
\def\nonewpage{\global\nonewpagetrue}

%% Begin twocolumns ==>>
\newdimen\checkforspace
\def\twocolumns{\par \penalty\z@
\checkforspace=\pagegoal
\advance\checkforspace-\pagetotal
\advance\checkforspace-\ht\partialpage
\advance\checkforspace-\dp\partialpage
\advance\checkforspace by -2\ht\dbltopins
\advance\checkforspace by -2\dp\dbltopins
\advance\checkforspace by -2\ht\dblbotins
\advance\checkforspace by -2\dp\dblbotins
\ifdim\checkforspace<36pt\ifnonewpage\else\newpage\fi\fi
\begingroup
%
\expandafter\gdef\csname plate*\endcsname{\twocolplate}
\expandafter\gdef\csname endplate*\endcsname{\endtwocolfloat}
%
\expandafter\gdef\csname figure*\endcsname{\twocolfig}
\expandafter\gdef\csname endfigure*\endcsname{\endtwocolfloat}
%
\expandafter\gdef\csname table*\endcsname{\twocoltable}
\expandafter\gdef\csname endtable*\endcsname{\endtwocolfloat}
%
\let\figure\onecolfig
\let\plate\onecolplate
\let\table\onecoltable
%
\let\@makecol\new@makecol %%
%%      Confines new def of \@makecol to
%%      two column material, preserves normal topskip after twocolumns is
%%      finished.
%%
\output={\global\setbox\partialpage=\vbox{\unvbox\@cclv}}
\vskip1sp %%
\eject
  \vbadness10001 \hbadness5000
  \tolerance5000
\parskip  \saveparskip %plus .01pt %
%\ifjdraft
%\baselineskip=28pt\fi
   \pretolerance\m@ne %keeps TeX from trying to make up paragraph
   %  without discretionary hyphens
  \advance\@colroom-\ht\partialpage
  \advance\@colroom-\dp\partialpage
\vsize=\@colroom
% Changed to \@colroom rather than 2\@colroom. Each column is output
% separately so we can find out if there are top or bottom inserts
% in first column.
%%
  \hsize\textwidth \advance\hsize-\columnsep
  \divide\hsize by 2
  \linewidth\hsize %% particular to LaTeX
  \columnwidth=\hsize
%% OR, you can substitute these lines:
%\hsize=\mycolumnwidth
%\linewidth=\hsize
%%
%% Output:
%  Set output to firstcolout, which checks to see if there are inserts
%  then doubles the vsize and send material back into input stream, then
%  sets the next output routine to \makeuppage
%
\output={\firstcolout}
\vskip2sp\ignorespaces}


\def\galleycolumns{\par \penalty\z@
\checkforspace=\pagegoal
\advance\checkforspace-\pagetotal
\advance\checkforspace-\ht\partialpage
\advance\checkforspace-\dp\partialpage
\advance\checkforspace by -2\ht\dbltopins
\advance\checkforspace by -2\dp\dbltopins
\advance\checkforspace by -2\ht\dblbotins
\advance\checkforspace by -2\dp\dblbotins
\ifdim\checkforspace<36pt\newpage\fi
\begingroup
\expandafter\gdef\csname plate*\endcsname{\twocolplate}
\expandafter\gdef\csname endplate*\endcsname{\endtwocolfloat}
%
\expandafter\gdef\csname figure*\endcsname{\twocolfig}
\expandafter\gdef\csname endfigure*\endcsname{\endtwocolfloat}
%
\expandafter\gdef\csname table*\endcsname{\twocoltable}
\expandafter\gdef\csname endtable*\endcsname{\endtwocolfloat}
%
\let\figure\onecolfig
\let\plate\onecolplate
\let\table\onecoltable
%
\global\firstsectionfalse
\hsize=\mycolumnwidth
\linewidth=\mycolumnwidth
  \vbadness10001
\hbadness5000
  \tolerance5000
\widowpenalty=0 \clubpenalty=0
\parskip  \saveparskip %plus .0001pt %
\vskip1sp\ignorespaces}

\def\draftcolumns{\par \penalty\z@
\begingroup
\expandafter\gdef\csname plate*\endcsname{\twocolplate}
\expandafter\gdef\csname endplate*\endcsname{\endtwocolfloat}
%
\expandafter\gdef\csname figure*\endcsname{\twocolfig}
\expandafter\gdef\csname endfigure*\endcsname{\endtwocolfloat}
%
\expandafter\def\csname table*\endcsname{\twocoltable}
\expandafter\def\csname endtable*\endcsname{\endtwocolfloat}
%
\let\figure\onecolfig
\let\plate\onecolplate
\let\table\onecoltable
%
\marginparsep=4pt
\oddsidemargin-.25in
\evensidemargin-.25in
\advance\textwidth-.5in
\hsize=\textwidth
\linewidth=\textwidth
  \vbadness10001 \hbadness5000
  \tolerance5000
\parskip  \saveparskip
\vskip1sp\ignorespaces}

\newbox\savetopinsert
\newbox\savebotinsert
\newbox\secondsavetopinsert
\newbox\secondsavebotinsert
\newbox\firstcolbox
\newbox\secondcolbox
\newbox\thefirstbox
\newbox\thesecondbox

\newdimen\firstcol
\newdimen\secondcol
\newdimen\checkheight

\newcount\loopnum
\newcount\firstcolsavefignum

\def\firstcolout{%
%% The point of having firstcolout is to find out
%% which figures/tables have been entered in the first column:
\global\firstcolsavefignum=\figandtabnumber
\global\advance\vsize by\@colroom
\topskip=11pt %% ??
       \unvbox\@cclv
       \penalty\outputpenalty
\global\output={\makeuppage}}

%% This is where the floats and text are put together to make page:

\newif\iffirstpage
\newbox\footnotebox

\newdimen\ColUsed
\newdimen\FigSpace
\newbox\testdblbotins

\def\makeuppage{%
%%
\FigSpace=.9\@colroom %%
\advance\FigSpace by-\ht\testdblbotins
\advance\FigSpace by-\dp\testdblbotins
\advance\FigSpace by-\ht\dbltopins
\advance\FigSpace by-\dp\dbltopins
%
%% Save top inserts from first column: +++
\global\setbox\savetopinsert=\vbox{%
% write top figures and tables into this box
\redefiningtrue
\global\loopnum=0 \global\checkheight=0pt
\loop\ifnum\loopnum<\firstcolsavefignum
 %
\expandafter\ifx\csname topfloat\the\loopnum\endcsname\empty
\else
\expandafter\ifx\csname topfloat\the\loopnum\endcsname\relax
\else
\global\advance\checkheight by
\expandafter\csname totalht\the\loopnum\endcsname\relax
\fi\fi
\ifdim\checkheight>\FigSpace \relax
\global\loopnum=10000 % jump out of loop
%
\else
\csname topfloat\the\loopnum\endcsname
\fi
  %
\global\advance\loopnum by1
\repeat\relax
}
\ColUsed=\FigSpace %%
\advance\ColUsed by-\ht\savetopinsert
\advance\ColUsed by-\dp\savetopinsert
  %%
\global\setbox\savebotinsert=\vbox{%
  %% Write bottom figures and tables into this box
\redefiningtrue
\global\loopnum=0
\global\checkheight=0pt
\loop\ifnum\loopnum<\firstcolsavefignum
\expandafter\ifx\csname botfloat\the\loopnum\endcsname\empty
\else
\expandafter\ifx\csname botfloat\the\loopnum\endcsname\relax
\else
\global\advance\checkheight by
\expandafter\csname totalht\the\loopnum\endcsname\relax
\fi\fi
\ifdim\checkheight>\ColUsed
\global\loopnum=10000 % jump out of loop
\else%
\csname botfloat\the\loopnum\endcsname
\global\expandafter\expandafter\let\csname botfloat\the\loopnum\endcsname\relax
%% how come this reset is here and not in similar places????
\fi
  %
\global\advance\loopnum by1
\repeat
} %
  %
%
\iffirstpage\else
\global\setbox\secondsavetopinsert=\vbox{%
%% Write top figures and tables into this box
%
\global\loopnum=0 \global\checkheight=0pt
\redefiningtrue
\loop\ifnum\loopnum<\figandtabnumber %% ?
%
\expandafter\ifx\csname topfloat\the\loopnum\endcsname\empty
\else
\expandafter\ifx\csname topfloat\the\loopnum\endcsname\relax
\else
\global\advance\checkheight by
\expandafter\csname totalht\the\loopnum\endcsname\relax
\fi\fi
\ifdim\checkheight>\FigSpace
\global\loopnum=10000 % jump out of loop
%
\else%
\csname topfloat\the\loopnum\endcsname
\fi
%
\global\advance\loopnum by1\repeat
%% one more, because it doesn't stop when number is =, only when it
%% is less than.
\global\advance\loopnum by1
\csname topfloat\the\loopnum\endcsname
}
\fi
%
\ColUsed=\FigSpace %%
\advance\ColUsed by-\ht\secondsavetopinsert
\advance\ColUsed by-\dp\secondsavetopinsert
%
\iffirstpage\global\firstpagefalse\else
\global\setbox\secondsavebotinsert=\vbox{%
%% Write bottom figures and tables into this box
%
\redefiningtrue
\global\loopnum=0 \global\checkheight=0pt
\loop\ifnum\loopnum<\figandtabnumber
\expandafter\ifx\csname botfloat\the\loopnum\endcsname\empty
\else
\expandafter\ifx\csname botfloat\the\loopnum\endcsname\relax
\else
\global\advance\checkheight by
\expandafter\csname totalht\the\loopnum\endcsname\relax
\fi\fi
\ifdim\checkheight>\ColUsed
\global\loopnum=10000 % jump out of loop
%
\else%
\csname botfloat\the\loopnum\endcsname
\fi
%
\global\advance\loopnum by1\repeat
%% one more, because it doesn't stop when number is =, only when it
%% is less than.
\global\advance\loopnum by1
\csname botfloat\the\loopnum\endcsname
}
%
\global\dp\secondsavebotinsert=2.5pt
\global\dp\savebotinsert=2.5pt
\fi
%
   \splittopskip=\topskip
   \splitmaxdepth\maxdepth
   \dimen@\@colroom
%% Leave space if there is a two-column wide bottom insertion:
%
\global\setbox\dblbotins=\vbox{%
\hsize=\textwidth
\linewidth=\textwidth
\captionwidth=\widecaptionwidth
\unvbox\dblbotins %% avoid eradicating this if it has anything in it
%% Write dblspan bottom figures and tables into this box
\redefiningtrue
\global\dblloopnum=0
\loop\ifnum\dblloopnum<\savedblfigandtabnumber %%
\csname dblbotfloat\the\dblloopnum\endcsname
\global\advance\dblloopnum by1\repeat
\csname dblbotfloat\the\dblloopnum\endcsname
}
   \ifvoid\dblbotins \else
      \advance\dimen@-\ht\dblbotins
      \advance\dimen@-\dp\dblbotins
\fi
%
%% Leave space if there is a two-column wide top insertion:
   \ifvoid\dbltopins\else
      \advance\dimen@-\ht\dbltopins
      \advance\dimen@-\dp\dbltopins
\fi
%%
%% These changes are made to cut the column size down if we need
%% to fit an insert into the column. This allows the right and
%% left column to have differing amounts of text cut from box 255,
%% called box \@cclv in LaTeX. See \vsplit below.
\firstcol=\dimen@
\ifvoid\savetopinsert\else
\ifdim\ht\savetopinsert>0pt
\advance\firstcol by-\ht\savetopinsert
\advance\firstcol by-\dp\savetopinsert\fi\fi
\ifvoid\savebotinsert\else
\ifdim\ht\savebotinsert>0pt
\advance\firstcol by-\ht\savebotinsert
\advance\firstcol by-\dp\savebotinsert\fi
\fi%
%%% KLUDGE below
\advance\firstcol by-3pt %% We need this to avoid
                         %% error messages about overfull boxes.
%
\secondcol=\dimen@
%%
\ifvoid\secondsavetopinsert\else
\ifdim\ht\secondsavetopinsert>0pt
\advance\secondcol by-\ht\secondsavetopinsert
\advance\secondcol by-\dp\secondsavetopinsert\fi\fi
\ifvoid\secondsavebotinsert\else
\ifdim\ht\secondsavebotinsert>0pt
\advance\secondcol by-\ht\secondsavebotinsert
\advance\secondcol by-\dp\secondsavebotinsert\fi\fi
\ifvoid\footins\else
\ifdim\ht\footins>0pt
\global\setbox\footnotebox=\vbox{\vskip\skip\footins
\footnoterule\unvbox\footins\vskip1sp}
\advance\secondcol by-\ht\footnotebox
\advance\secondcol by-\dp\footnotebox\fi\fi
%%% KLUDGE below
\advance\secondcol by-3pt %% We need this to avoid
                          %% error messages about overfull boxes.
%
%% Cutting the amount of text that will fit from box255:
   \splittopskip=\topskip %%
   \splitmaxdepth\maxdepth
\setbox\thefirstbox \vsplit\@cclv to\firstcol
\setbox\thesecondbox \vsplit\@cclv to\secondcol
%
%% Making a text box that includes the inserts, column to the left:
\setbox\firstcolbox\vbox to\dimen@{%
\ifvoid\savetopinsert\else\unvbox\savetopinsert\fi%
\box\thefirstbox%
\ifvoid\savebotinsert\else\vfill\unvbox\savebotinsert\fi%
}
%
%% Making a text box that includes the inserts, column to the right:
\setbox\secondcolbox\vbox to\dimen@{%
\ifvoid\secondsavetopinsert\else\unvbox\secondsavetopinsert\fi%
\box\thesecondbox%%
\ifvoid\secondsavebotinsert\else\vfill\unvbox\secondsavebotinsert\vskip1sp\fi%
\ifvoid\footnotebox\else\unvbox\footnotebox\fi%
}
% Put back any text material that is not used:
  \ifvoid\@cclv \else
       \unvbox\@cclv
       \penalty\outputpenalty\fi
   \setbox\@cclv\vbox{\page@sofar}%
   \@makecol\@outputpage
   \global\@colroom\@colht
% Changed this so that each column will be output separately,
% so we have a chance to see if there are any insertions in the first
% column.
%
%% Do Page Inserts here, so that if there are any two col top inserts
%% They will be numbered after Page Inserts.
%%
\expandafter\ifx\csname pagefloat\the\pageloopnum\endcsname\relax
\else
\expandafter\ifx\csname pagefloat\the\pageloopnum\endcsname\empty
\else
\insertpage\fi\fi
%
%** doing this now gets the numbers right.
\global\setbox\dbltopins=\vbox{%
\hsize=\textwidth
\linewidth=\textwidth
\captionwidth=\widecaptionwidth
\unvbox\dbltopins
%% Write dblspan bottom figures and tables into this box
\redefiningtrue %
\global\dblloopnum=0
\loop\ifnum\dblloopnum<\savedblfigandtabnumber
\csname dbltopfloat\the\dblloopnum\endcsname
\global\advance\dblloopnum by1\repeat
\csname dbltopfloat\the\dblloopnum\endcsname%
                                %Save one for next page?
%\vskip24pt
}
%**
%
%% dblfigandtabnumber rather than savedblfigandtabnumber here
\global\setbox\testdblbotins=\vbox{%
\let\c@figure\bogus
\let\c@table\bogus
\let\c@plate\bogus
\hsize=\textwidth
\linewidth=\textwidth
\captionwidth=\widecaptionwidth
\copy\dblbotins %% avoid eradicating this if it has anything in it
%% Write dblspan bottom figures and tables into this box
\redefiningfalse
\global\dblloopnum=0
\loop\ifnum\dblloopnum<\dblfigandtabnumber %%
\csname dblbotfloat\the\dblloopnum\endcsname
\global\advance\dblloopnum by1\repeat
\csname dblbotfloat\the\dblloopnum\endcsname
}
%
\ifbalancing
\global\balancingfalse
\global\vsize=2\@colroom
\else
\global\vsize\@colroom
\global\advance\vsize -\ht\dbltopins
\global\advance\vsize -\dp\dbltopins
\global\advance\vsize -\ht\testdblbotins
\global\advance\vsize -\dp\testdblbotins
\global\output={\firstcolout}%
\fi
}


\newcount\pageloopnum

\def\secondoutput{{\makeuppage }}

\newbox\endcolsavetopinsert
\newbox\endcolsavebotinsert
\newbox\testpagesize
\newif\ifbalancing

\newdimen\savelastskip
%%+++
\def\endtwocolumns{%
%\par\penalty0
\splittopskip=\topskip
\splitmaxdepth\maxdepth
\setboxesandredefine
\ifdim\ht\endcolsavetopinsert>1pt
\unvbox\endcolsavetopinsert%
\fi
\ifdim\ht\endcolsavebotinsert>1pt
\unvbox\endcolsavebotinsert%
\fi
\par\penalty\z@
\output={\global\setbox\testpagesize=\vbox{%
\unvbox\@cclv\ifdim\savelastskip>0pt \vskip-\savelastskip\fi}}\eject
 %%
 %% We reset output routine below to \latexoutput
 %% since we had to set multicolout to be the global output routine.
 %%
\dimen0=\ht\testpagesize
\advance\dimen0 by\dp\testpagesize
\unvbox\testpagesize %%
%% Above needed to see if we have more than one page worth of text.
%%
\advance\dimen0 by \topskip
\ifdim\dimen0>2\@colroom
%\typeout{^^J^^J more than 2 colroom^^J^^J}
 %% More than one page of text:
 %%
\balancingtrue %% need this to avoid turning on firstcolout output routine.
\global\output={\settwocolboxes\makeuppage
\global\output={\setboxesandredefine
\balance@columns\global\output={\latexoutput}}}
\else
\settwocolboxes
\advance\dimen0 by 2\ht\dbltopins
\advance\dimen0 by 2\dp\dbltopins
\ifdim\dimen0>2\@colroom
%\typeout{^^J^^J Second more than 2 colroom^^J^^J}
 %%
\balancingtrue %% need this to avoid turning on firstcolout output routine.
\global\output={\makeuppage
\global\output={\balance@columns\global\output={\latexoutput}}}
\else
%\typeout{^^J^^J NOT more than 2 colroom^^J^^J}
\balancingtrue %% need this to avoid turning on firstcolout output routine.
\global\output={\balance@columns\global\output={\latexoutput}}\fi\fi
\par
\eject
\endgroup
\par\penalty\z@\relax
\hsize=\textwidth
}

%% to get rid of spurious error message when triple col is used:
%\def\@checkend#1{\def\reserved@a{#1}\ifx
%      \reserved@a\@currenvir \else\iftriplecol
%\else\@badend{#1}\fi\fi}

\def\endgalley{\vskip1sp\endgroup\newpage
\vspace*{1in}
\unvbox\figinsert}

\def\enddraft{\vskip1sp\endgroup
\ifdim\ht\figinsert <2pt\else
\newpage
\vspace*{1in}
\unvbox\figinsert\fi}

\def\setboxesandredefine{%
\global\setbox\dbltopins=\vbox{%
\hsize=\textwidth
\linewidth=\textwidth
\captionwidth=\widecaptionwidth
\unvbox\dbltopins
%% Write dblspan bottom figures and tables into this box
\redefiningtrue %
\global\dblloopnum=0
\loop\ifnum\dblloopnum<\dblfigandtabnumber
\csname dbltopfloat\the\dblloopnum\endcsname
\global\advance\dblloopnum by1\repeat
\csname dbltopfloat\the\dblloopnum\endcsname%
%                           %Save one for next page?
\vskip3pt%% kludge
\vskip-\intextfloatskip
}
%
\global\setbox\endcolsavetopinsert=\vbox{%
%% dropping figures and tables into this box
\loopnum=0
\redefiningtrue
\loop\ifnum\loopnum<\figandtabnumber
\csname topfloat\the\loopnum\endcsname
\global\advance\loopnum by1\repeat
\csname topfloat\the\loopnum\endcsname%
}
%
\global\setbox\endcolsavebotinsert=\vbox{%
%% drop figures and tables into this box
\global\loopnum=0
\redefiningtrue
\loop\ifnum\loopnum<\figandtabnumber
\csname botfloat\the\loopnum\endcsname
\global\advance\loopnum by1\repeat
\csname botfloat\the\loopnum\endcsname}
%
\global\setbox\dblbotins=\vbox{%
\hsize=\textwidth
\linewidth=\textwidth
\captionwidth=\widecaptionwidth
\unvbox\dblbotins %% avoid eradicating this if it has anything in it
% Write dblspan bottom figures and tables into this box
\redefiningtrue
\global\dblloopnum=0
\loop\ifnum\dblloopnum<\dblfigandtabnumber %%
\csname dblbotfloat\the\dblloopnum\endcsname
\global\advance\dblloopnum by1\repeat
\csname dblbotfloat\the\dblloopnum\endcsname}%
}%% end of setboxes and redefine

\def\settwocolboxes{%
\global\setbox\dbltopins=\vbox{%
\hsize=\textwidth
\linewidth=\textwidth
\captionwidth=\widecaptionwidth
\unvbox\dbltopins
% Write dblspan bottom figures and tables into this box
\redefiningtrue %
\global\dblloopnum=0
\loop\ifnum\dblloopnum<\dblfigandtabnumber
\csname dbltopfloat\the\dblloopnum\endcsname
\global\advance\dblloopnum by1\repeat
\csname dbltopfloat\the\dblloopnum\endcsname%
%
%\global\setbox\dblbotins=\vbox
\hsize=\textwidth
\linewidth=\textwidth
\captionwidth=\widecaptionwidth
\unvbox\dblbotins %% avoid eradicating this if it has anything in it
% Write dblspan bottom figures and tables into this box
\redefiningtrue
\global\dblloopnum=0
\loop\ifnum\dblloopnum<\dblfigandtabnumber %%
\csname dblbotfloat\the\dblloopnum\endcsname
\global\advance\dblloopnum by1\repeat
\csname dblbotfloat\the\dblloopnum\endcsname%
\vskip\intextfloatskip%% new++
}}

\def\setonecolboxesandredefine{%
\global\setbox\endcolsavetopinsert=\vbox{%
%% dropping figures and tables into this box
\loopnum=0
\redefiningtrue
\loop\ifnum\loopnum<\figandtabnumber
\csname topfloat\the\loopnum\endcsname
\expandafter\ifx\csname topfloat\the\loopnum\endcsname\relax\else
\vskip12pt\goodbreak\fi
\global\advance\loopnum by1\repeat
\csname topfloat\the\loopnum\endcsname%
}
%
\global\setbox\endcolsavebotinsert=\vbox{%
%% drop figures and tables into this box
\global\loopnum=0
\redefiningtrue
\loop\ifnum\loopnum<\figandtabnumber
\csname botfloat\the\loopnum\endcsname
% we don't want a space here because it is at the bottom of a column:
%\expandafter\ifx\csname botfloat\the\loopnum\endcsname\relax\else
%\vskip12pt\goodbreak\fi
\global\advance\loopnum by1\repeat
\csname botfloat\the\loopnum\endcsname%
}
%
}%% end of setonecol boxes and redefine

\newbox\partialpage
\def\process@cols#1#2{\count@#1\relax
     \loop #2%
     \advance\count@\tw@
     \ifnum\count@<4
   \repeat}

%%%   Version for balanced column output.

%% this version to get topskip right on balanced page at end of article.

\def\Bpage@sofar{
\unvbox\partialpage
%% major kludge! but it works ==>>
\null
\vskip-\topskip
\vskip-12pt
%% <<==
   \process@cols\z@{\wd\count@\hsize}%
   \hbox to\textwidth{%
     \process@cols\tw@{\box\count@
       \hss\vrule\@width\columnseprule\hss}%
     \box\z@}}

%%% AH: Version for two column output. %%
\def\page@sofar{\unvbox\partialpage
\wd\firstcolbox\hsize
\wd\secondcolbox\hsize
\hbox to\textwidth{%
\box\firstcolbox
   \hss\vrule\@width\columnseprule\hss
\box\secondcolbox}%%
}


%% Variant on \@outputpage to use for floating page insert:
\def\insertpage{\begingroup\catcode`\ =10
     \let\-\@dischyph \let\'\@acci \let\`\@accii \let\=\@acciii
     \if@twoside
       \ifodd\count\z@ \let\@thehead\@oddhead \let\@thefoot\@oddfoot
            \let\@themargin\oddsidemargin
          \else \let\@thehead\@evenhead
          \let\@thefoot\@evenfoot \let\@themargin\evensidemargin
     \fi\fi
     \shipout
     \vbox{\normalsize \baselineskip\z@ \lineskip\z@
           \let\par\@@par %%
           \vskip \topmargin \moveright\@themargin
           \vbox{\setbox\@tempboxa
                   \vbox to\headheight{\vfil \hbox to\textwidth
                                       {\let\label\@gobble \let\index\@gobble
                                         \@thehead}} %%
                 \dp\@tempboxa\z@
                 \box\@tempboxa
                 \vskip \headsep
%% change from \@outputpage below
\vbox to\textheight{\vfil
\hsize=\textwidth
\linewidth=\textwidth
\csname pagefloat\the\pageloopnum \endcsname
}
                 \baselineskip\footskip
                 \hbox to\textwidth{\let\label\@gobble
                           \let\index\@gobble  %%
                           \@thefoot}}}\global\@colht\textheight
           \endgroup\stepcounter{page}\let\firstmark\botmark
\global\advance\pageloopnum by1
\expandafter\ifx\csname pagefloat\the\pageloopnum\endcsname\relax\else%
\insertpage\fi%
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\def\balance@columns{% can't throw in figs here because we
                % aren't doing eject, so they will be lost
\ifdim\ht\dbltopins>1pt
  \advance\@colroom-\ht\dbltopins
  \advance\@colroom-\dp\dbltopins
\advance\@colroom-\baselineskip
\fi
\ifdim\ht\dblbotins>1pt
 \advance\@colroom-\ht\dblbotins
  \advance\@colroom-\dp\dblbotins
\fi
  \setbox\z@\vbox{\unvbox\@cclv}
\dimen@\ht\z@
%\advance\dimen@\dp\z@
\advance\dimen@ 2\topskip
   \advance\dimen@-2\baselineskip %% ?? was 2\baselineskip, 4 gives better
                                  %topskip
   \divide\dimen@2%
%
{\vbadness\@M
   \splittopskip=\topskip
   \loop{\process@cols\@ne{\global\setbox\count@
                             \box\voidb@x}}%
     \global\setbox\@ne\copy\z@
   {\process@cols\thr@@{\global\setbox\count@
                     \vsplit\@ne to\dimen@}}%
    \ifdim\ht\@ne >\ht\thr@@
    \global\advance\dimen@\p@
    \repeat}
%
   \dimen@\ht\thr@@
   \process@cols\z@{\@tempcnta\count@
        \advance\@tempcnta\@ne
        \setbox\count@\vtop to\dimen@ %%
           {\unvbox\@tempcnta}}%
   \global\vsize\@colroom
   \global\advance\vsize\ht\partialpage
   \Bpage@sofar}

\@ifundefined{emergencystretch}
     {\newdimen\emergencystretch}{}

\emergencystretch 1pt

\def\setemergencystretch#1#2{%
   \emergencystretch 4pt
   \multiply\emergencystretch#1}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Figure and table captions in two column text and spanning both
%%% columns.

%%%%%%%%%%%%%%%%%%%%%%%%%
%% Single column inserts:

\let\savecaption\caption
\let\xsavecaption\savecaption

\def\onecolfig{\ifvmode\else\unskip\fi
\global\captypefigtrue
\global\captypeplatefalse
\@ifnextchar[\lookforposition{\lookforposition[e]}}

\def\onecolplate{\ifvmode\else\unskip\fi
\global\advance\c@plate -1\relax
\global\captypeplatetrue
\global\captypefigfalse
\@ifnextchar[\lookforposition{\lookforposition[e]}}

\def\onecoltable{\ifvmode\else\unskip\vskip1pt\fi\tabletextsize
\global\captypefigfalse
\global\captypeplatefalse
\@ifnextchar[\lookforposition{\lookforposition[e]}
}

%% This is used in case we get more than one letter, i.e.,
%% \begin{figure}[htp]. This macro picks up only the first letter.

\def\defone#1#2*{\def\one{#1}}

\newcount\figandtabnumber
\newbox\figandtabbox
\newif\ifcaptypefig
\newif\ifcaptypeplate

\long\def\catchcaption#1{%
\ifcaptypefig%
\expandafter\gdef\csname caption\the\figandtabnumber\endcsname{%
\def\@captype{figure}\savecaption{#1}}%
\else%
\ifcaptypeplate
\expandafter\gdef\csname caption\the\figandtabnumber\endcsname{%
\def\@captype{plate}\savecaption{#1}}%
\else
\expandafter\gdef\csname caption\the\figandtabnumber\endcsname{%
\def\@captype{table}\savecaption{#1}}%
\fi\fi}

\long\def\dblcatchcaption#1{%
\ifcaptypefig%
\expandafter\gdef\csname dblcaption\the\dblfigandtabnumber\endcsname{%
\def\@captype{figure}\savecaption{#1}}%
\else%
\ifcaptypeplate
\expandafter\gdef\csname dblcaption\the\dblfigandtabnumber\endcsname{%
\def\@captype{plate}\savecaption{#1}}%
\else
\expandafter\gdef\csname dblcaption\the\dblfigandtabnumber\endcsname{%
\def\@captype{table}\savecaption{#1}}%
\fi\fi}

\long\def\pcatchcaption#1{%
\ifcaptypefig%
\expandafter\gdef\csname pcaption\the\pagefloatnumber\endcsname{%
\def\@captype{figure}\savecaption{#1}}%
\else%
\ifcaptypeplate
\expandafter\gdef\csname pcaption\the\pagefloatnumber\endcsname{%
\def\@captype{plate}\savecaption{#1}}%
\else
\expandafter\gdef\csname pcaption\the\pagefloatnumber\endcsname{%
\def\@captype{table}\savecaption{#1}}%
\fi\fi}
%%%

\long\def\catchletteredcaption#1#2{%
\ifcaptypefig%
\expandafter\gdef\csname letteredcaption\the\figandtabnumber\endcsname{%
\def\@captype{figure}\letteredcaption{#1}{#2}}%
\else%
\ifcaptypeplate
\expandafter\gdef\csname letteredcaption\the\figandtabnumber\endcsname{%
\def\@captype{plate}\letteredcaption{#1}{#2}}%
\else
\expandafter\gdef\csname letteredcaption\the\figandtabnumber\endcsname{%
\def\@captype{table}\letteredcaption{#1}{#2}}%
\fi\fi}


\long\def\dblcatchletteredcaption#1#2{%
\ifcaptypefig%
\expandafter\gdef\csname dblletteredcaption\the\dblfigandtabnumber\endcsname{%
\def\@captype{figure}%
\letteredcaption{#1}{#2}}%
\else%
\ifcaptypeplate
\expandafter\gdef\csname dblletteredcaption\the\dblfigandtabnumber\endcsname{%
\def\@captype{plate}%
\letteredcaption{#1}{#2}}%
\else
\expandafter\gdef\csname dblletteredcaption\the\dblfigandtabnumber\endcsname{%
\def\@captype{table}%
\letteredcaption{#1}{#2}}%
\fi\fi}

\long\def\pcatchletteredcaption#1#2{%
\ifcaptypefig%
\expandafter\gdef\csname pletteredcaption\the\pagefloatnumber\endcsname{%
\def\@captype{figure}%
\letteredcaption{#1}{#2}}%
\else%
\ifcaptypeplate%
\expandafter\gdef\csname pletteredcaption\the\pagefloatnumber\endcsname{%
\def\@captype{plate}%
\letteredcaption{#1}{#2}}%
\else
\expandafter\gdef\csname pletteredcaption\the\pagefloatnumber\endcsname{%
\def\@captype{table}%
\letteredcaption{#1}{#2}}%
\fi\fi}

%%%
\long\def\catchcontinuedcaption{%
\ifcaptypefig%
\expandafter\gdef\csname continuedcaption\the\figandtabnumber\endcsname{%
\def\@captype{figure}\continuedcaption}%
\else%
\ifcaptypeplate
\expandafter\gdef\csname continuedcaption\the\figandtabnumber\endcsname{%
\def\@captype{plate}\continuedcaption}%
\else
\expandafter\gdef\csname continuedcaption\the\figandtabnumber\endcsname{%
\def\@captype{table}\continuedcaption}%
\fi\fi}

\long\def\dblcatchcontinuedcaption{%
\ifcaptypefig%
\expandafter\gdef\csname dblcontinuedcaption\the\dblfigandtabnumber\endcsname{%
\def\@captype{figure}\continuedcaption}%
\else%
\ifcaptypeplate
\expandafter\gdef\csname dblcontinuedcaption\the\dblfigandtabnumber\endcsname{%
\def\@captype{plate}\continuedcaption}%
\else
\expandafter\gdef\csname dblcontinuedcaption\the\dblfigandtabnumber\endcsname{%
\def\@captype{table}\continuedcaption}%
\fi\fi}

\long\def\pcatchcontinuedcaption{%
\ifcaptypefig%
\expandafter\gdef\csname pcontinuedcaption\the\pagefloatnumber\endcsname{%
\def\@captype{figure}\continuedcaption}%
\else%
\ifcaptypeplate
\expandafter\gdef\csname pcontinuedcaption\the\pagefloatnumber\endcsname{%
\def\@captype{plate}\continuedcaption}%
\else
\expandafter\gdef\csname pcontinuedcaption\the\pagefloatnumber\endcsname{%
\def\@captype{table}\continuedcaption}%
\fi\fi}

%%%
\long\def\catchletteredcontinuedcaption#1{%
\ifcaptypefig%
\expandafter\gdef\csname letteredcontcaption\the\figandtabnumber\endcsname{%
\def\@captype{figure}\letteredcontinuedcaption{#1}}%
\else%
\ifcaptypeplate%
\expandafter\gdef\csname letteredcontcaption\the\figandtabnumber\endcsname{%
\def\@captype{plate}\letteredcontinuedcaption{#1}}%
\else
\expandafter\gdef\csname letteredcontcaption\the\figandtabnumber\endcsname{%
\def\@captype{table}\letteredcontinuedcaption{#1}}%
\fi\fi}


\long\def\dblcatchletteredcontinuedcaption#1{%
\ifcaptypefig%
\expandafter%
\gdef\csname dblletteredcontcaption\the\dblfigandtabnumber\endcsname{%
\def\@captype{figure}\letteredcontinuedcaption{#1}}%
\else%
\ifcaptypeplate
\expandafter%
\gdef\csname dblletteredcontcaption\the\dblfigandtabnumber\endcsname{%
\def\@captype{plate}\letteredcontinuedcaption{#1}}%
\else
\expandafter%
\gdef\csname dblletteredcontcaption\the\dblfigandtabnumber\endcsname{%
\def\@captype{table}\letteredcontinuedcaption{#1}}%
\fi\fi}

\long\def\pcatchletteredcontinuedcaption#1{%
\ifcaptypefig%
\expandafter%
\gdef\csname pletteredcontcaption\the\pagefloatnumber\endcsname{%
\def\@captype{figure}\letteredcontinuedcaption{#1}}%
\else%
\ifcaptypeplate%
\expandafter%
\gdef\csname pletteredcontcaption\the\pagefloatnumber\endcsname{%
\def\@captype{plate}\letteredcontinuedcaption{#1}}%
\else
\expandafter%
\gdef\csname pletteredcontcaption\the\pagefloatnumber\endcsname{%
\def\@captype{table}\letteredcontinuedcaption{#1}}%
\fi\fi}

%%%


\newcount\pagefloatnumber

\let\savelabel\label
\def\catchlabel#1{%
\expandafter\gdef\csname label\the\figandtabnumber\endcsname{\savelabel{#1}}}

\def\pcatchlabel#1{%
\expandafter\gdef\csname plabel\the\pagefloatnumber\endcsname{\savelabel{#1}}}

\def\dblcatchlabel#1{\expandafter%
\gdef\csname dbllabel\the\dblfigandtabnumber\endcsname{\savelabel{#1}}}

%%

\def\startonecolfloat{%
\expandafter\newbox\csname figandtabbox\the\figandtabnumber\endcsname%
\global\setbox\csname figandtabbox\the\figandtabnumber\endcsname=
\vbox\bgroup%
\ifcaptypefig\else
\ifcaptypeplate\def\@captype{plate}
\else
\def\@captype{table}\centering\tabletextsize\fi\fi%
\let\continuedcaption\catchcontinuedcaption
\let\letteredcaption\catchletteredcaption
\let\letteredcontinuedcaption\catchletteredcontinuedcaption
\let\caption\catchcaption
\let\label\catchlabel}

\def\xtopfloat{topfloat}
\def\xbotfloat{botfloat}

\newif\ifredefining


\newbox\sizeoffloat

\def\endonecolfloat{%
%\ifcaptypefig\else%
%\ifnobottomtabline\global\nobottomtablinefalse\vskip-6.5pt\else
%\vskip2pt
%\thetablelines\fi\fi%
%
\egroup%
\DonormalEndcol%
%% make room in column for this insert:
\setbox\sizeoffloat=\vbox{\let\unvbox\copy
\let\c@table\bogus
\let\c@figure\bogus
\redefiningfalse \loopnum=\figandtabnumber
\expandafter\csname \toporbotfloat\the\figandtabnumber\endcsname}%
%%
\expandafter\xdef\csname totalht\the\figandtabnumber\endcsname{%
\the\ht\sizeoffloat}% <=== use before \sizeoffloat is unboxed
\global\advance\figandtabnumber by1\relax}

\newcount\bogus
\def\DonormalEndcol{%
%% top float ==>
\ifx\toporbotfloat\xtopfloat%
%% figure ==>
  \ifcaptypefig%
  \expandafter\gdef\csname topfloat\the\figandtabnumber\endcsname{%
  \vbox{\vskip\PushOneColTopFig%
  \unvbox\csname figandtabbox\the\loopnum\endcsname%
  \vskip\abovefigcaptionskip%
  \csname caption\the\loopnum\endcsname%
  \csname letteredcaption\the\loopnum\endcsname%
  \csname continuedcaption\the\loopnum\endcsname%
  \csname letteredcontcaption\the\loopnum\endcsname}%
  \vskip\intextfloatskip%%
  \vskip-4pt %% probably an artifact of topskip??
  \ifredefining%
  \csname label\the\loopnum\endcsname%
  \expandafter\gdef\csname topfloat\the\loopnum\endcsname{}\fi}%
\else%
%% plate ==>
  \ifcaptypeplate%
  \expandafter\gdef\csname topfloat\the\figandtabnumber\endcsname{%
  \vbox{\vskip\PushOneColTopFig%
  \unvbox\csname figandtabbox\the\loopnum\endcsname
  \vskip\abovefigcaptionskip
  \csname caption\the\loopnum\endcsname
  \csname letteredcaption\the\loopnum\endcsname
  \csname continuedcaption\the\loopnum\endcsname
  \csname letteredcontcaption\the\loopnum\endcsname}
  \vskip\intextfloatskip %%
  \vskip-4pt %% probably an artifact of topskip??
  \ifredefining
  \csname label\the\loopnum\endcsname
  \expandafter\gdef\csname topfloat\the\loopnum\endcsname{}\fi}%
%
\else% table ==>
 \expandafter\gdef\csname topfloat\the\figandtabnumber\endcsname{%
 \vbox{\vskip\PushOneColTopTab %%
 \csname caption\the\loopnum\endcsname
  \csname letteredcaption\the\loopnum\endcsname
  \csname continuedcaption\the\loopnum\endcsname
  \csname letteredcontcaption\the\loopnum\endcsname
  \vskip\captionskip
  \unvbox\csname figandtabbox\the\loopnum\endcsname
}\vskip\intextfloatskip %% why don't we need this?
\vskip-10pt
\ifredefining
\csname label\the\loopnum\endcsname
\expandafter\gdef\csname topfloat\the\loopnum\endcsname{}\fi}
\fi\fi%
%
\else% bottom float
%
\ifcaptypefig
\expandafter\gdef\csname botfloat\the\figandtabnumber\endcsname{%
\vskip\intextfloatskip
\vbox{\unvbox\csname figandtabbox\the\loopnum\endcsname
\vskip\abovefigcaptionskip
  \csname caption\the\loopnum\endcsname
  \csname letteredcaption\the\loopnum\endcsname%
  \csname continuedcaption\the\loopnum\endcsname%
  \csname letteredcontcaption\the\loopnum\endcsname%
\vskip\PushOneColBotFig%%
}%
\ifredefining%
%\xdef\@currentlabel{\thefigure}%
\csname label\the\loopnum\endcsname
\expandafter\gdef\csname botfloat\the\loopnum\endcsname{}\fi}%
\else
\ifcaptypeplate
\expandafter\gdef\csname botfloat\the\figandtabnumber\endcsname{%
\vskip\intextfloatskip
\vbox{\unvbox\csname figandtabbox\the\loopnum\endcsname
\vskip\abovefigcaptionskip
  \csname caption\the\loopnum\endcsname
  \csname letteredcaption\the\loopnum\endcsname%
  \csname continuedcaption\the\loopnum\endcsname%
  \csname letteredcontcaption\the\loopnum\endcsname%
\vskip\PushOneColBotFig%%
}%
\ifredefining%
%\xdef\@currentlabel{\thefigure}%
\csname label\the\loopnum\endcsname
\expandafter\gdef\csname botfloat\the\loopnum\endcsname{}\fi}%
  \else% TABLE
\expandafter\gdef\csname botfloat\the\figandtabnumber\endcsname{%
  \vskip\intextfloatskip
\vbox{\csname caption\the\loopnum\endcsname
  \csname letteredcaption\the\loopnum\endcsname
  \csname continuedcaption\the\loopnum\endcsname
  \csname letteredcontcaption\the\loopnum\endcsname%
  \vskip.5\intextfloatskip
  \unvbox\csname figandtabbox\the\loopnum\endcsname%
\vskip\PushOneColBotTab
}%
\ifredefining%
\csname label\the\loopnum\endcsname
\expandafter\gdef\csname botfloat\the\loopnum\endcsname{}\fi}%
\fi\fi\fi}

%
\def\startpagefloat{%
\expandafter\newbox\csname pagebox\the\pagefloatnumber\endcsname%
\global\setbox\csname pagebox\the\pagefloatnumber\endcsname=\vbox\bgroup%
\hsize=\textwidth%
\linewidth=\textwidth%
\captionwidth=\widecaptionwidth
\let\label\pcatchlabel\ifcaptypefig\else\centering\fi%
\let\caption\pcatchcaption \let\letteredcaption\pcatchletteredcaption
\let\continuedcaption\pcatchcontinuedcaption
\let\letteredcontinuedcaption\pcatchletteredcontinuedcaption
}

\def\endpagefloat{\egroup
\DoNormalPage
\global\advance\pagefloatnumber by1
}

\def\DoNormalPage{\ifcaptypefig
%
\expandafter\gdef\csname pagefloat\the\pagefloatnumber\endcsname{%
\vbox{\hsize=\textwidth
\linewidth=\textwidth
\twocolfigortabtrue
\captionwidth=\widecaptionwidth
\unvbox\csname pagebox\the\pageloopnum\endcsname
  \vskip\captionskip
  \csname pcaption\the\pageloopnum\endcsname
  \csname pletteredcaption\the\pageloopnum\endcsname
  \csname pcontinuedcaption\the\pageloopnum\endcsname
  \csname pletteredcontcaption\the\pageloopnum\endcsname
}  \vskip\intextfloatskip %%
\xdef\@currentlabel{\thefigure}%
  \csname plabel\the\pageloopnum\endcsname
\expandafter\gdef\csname pagefloat\the\pageloopnum\endcsname{}}%
\else
\ifcaptypeplate
\expandafter\gdef\csname pagefloat\the\pagefloatnumber\endcsname{%
\vbox{\hsize=\textwidth
\linewidth=\textwidth
\twocolfigortabtrue
\captionwidth=\widecaptionwidth
\unvbox\csname pagebox\the\pageloopnum\endcsname
  \vskip\captionskip
  \csname pcaption\the\pageloopnum\endcsname
  \csname pletteredcaption\the\pageloopnum\endcsname
  \csname pcontinuedcaption\the\pageloopnum\endcsname
  \csname pletteredcontcaption\the\pageloopnum\endcsname
}  \vskip\intextfloatskip %%
\xdef\@currentlabel{\thefigure}%
  \csname plabel\the\pageloopnum\endcsname
\expandafter\gdef\csname pagefloat\the\pageloopnum\endcsname{}}%
\else
\expandafter\gdef\csname pagefloat\the\pagefloatnumber\endcsname{%
\vbox{%%
\hsize=\textwidth
\linewidth=\textwidth
\iflandscapetable
\dimen0=\textheight
\advance\dimen0 by-\textwidth
\hsize=\textheight
\linewidth=\textheight\fi%
\csname pcaption\the\pageloopnum\endcsname
\csname pletteredcaption\the\pageloopnum\endcsname
\csname pcontinuedcaption\the\pageloopnum\endcsname
\csname pletteredcontcaption\the\pageloopnum\endcsname
  \vskip\intextfloatskip
  \unvbox\csname pagebox\the\pageloopnum\endcsname}
  \vskip\intextfloatskip %%
\csname plabel\the\pageloopnum\endcsname
\expandafter\gdef\csname pagefloat\the\pageloopnum\endcsname{}}\fi\fi}

\newif\iffirstfighere
\global\firstfigheretrue

\newif\iffirsttabhere
\global\firsttabheretrue

\def\dofigmessage{\iffirstfighere\global\firstfigherefalse
\typeout{^^J
========================================================================
^^J
C A R E F U L !!!!!^^J
^^J
You have used \string\begin{figure}[h]\space !^^J
If there are any figures that appear earlier on the page^^J
they may be numbered incorrectly.^^J
^^J
Please check this and every other page on which you have used ^^J
\string\begin{figure}[h]\space !^^J
^^J
========================================================================
^^J}
\else
\typeout{^^J^^J
C A R E F U L !! \string\begin{figure}[h] used on this page !^^J^^J}
\fi}

\def\dotabmessage{\iffirsttabhere\global\firsttabherefalse
\typeout{^^J
========================================================================
^^J
C A R E F U L !!!!!^^J
^^J
You have used \string\begin{table}[h]\space !^^J
If there are any tables that appear earlier on the page^^J
they may be numbered incorrectly.^^J
^^J
Please check this and every other page on which you have used ^^J
\string\begin{table}[h]\space !^^J
^^J
========================================================================
^^J}
\else
\typeout{^^J^^J
C A R E F U L !! \string\begin{table}[h] used on this page !^^J^^J}
\fi}

\def\lookforposition[#1]{\defone#1*%
\let\go\startonecolfloat
\let\endtable\endonecolfloat
\let\endfigure\endonecolfloat
\let\endplate\endonecolfloat
\ifgalley
\if\one p \def\one{s}
\else
\def\one{i}\fi
%\else\ifappendon\def\one{h}\fi
\fi%
\ifjdraft
\expandafter\ifx\csname setkeys\endcsname\relax\else
\setkeys{Gin}{draft=false}\fi
\if\one p \def\one{s}
\else
\def\one{i}\fi
\else%\ifappendon\def\one{h}\fi
\fi%
\if\one h
\ifcaptypeplate\else
\ifcaptypefig
\dofigmessage
\else
\dotabmessage
\fi\fi%
\let\go\doonecolfighere
\let\endtable\endonecolfighere
\let\endfigure\endonecolfighere
\let\endplate\endonecolfighere
\else%
\if\one t%
    \gdef\toporbotfloat{topfloat}%
\else%
\if\one b%
    \gdef\toporbotfloat{botfloat}%
\else%
\if\one p%
\let\go\startpagefloat%
\let\endtable\endpagefloat%
\let\endfigure\endpagefloat%
\let\endplate\endpagefloat%
\else%
\if\one e% for either
   \ifdim\pagetotal>.6\pagegoal%
    \gdef\toporbotfloat{botfloat}%
    \else%
    \gdef\toporbotfloat{topfloat}%
    \fi%
\else%
\if\one i% for galley mode
\let\go\dofiginsert%
\let\endtable\endfiginsert%
\let\endfigure\endfiginsert%
\let\endplate\endfiginsert%
\else%
\if\one s% for special galley mode, for [p]
\let\go\dodblfiginsert%
\let\endtable\endspfiginsert%
\let\endfigure\endspfiginsert%
\let\endplate\endspfiginsert%
\else%
    \doerr%
    \ifdim\pagetotal>.5\pagegoal%
    \gdef\toporbotfloat{botfloat}%
    \else%
    \gdef\toporbotfloat{topfloat}%
    \fi%
\fi\fi\fi\fi\fi\fi\fi\go}

\newinsert\figinsert
\skip\figinsert=0pt % space added when figinsert is used
\count\figinsert=0 % insert magnification factor (1 to 1)
\dimen\figinsert=\maxdimen % maximum space for figures

\def\dofiginsert{\setbox0=\vbox\bgroup
\ifcaptypefig\def\@captype{figure}\else
\ifcaptypeplate\def\@captype{plate}\else
\def\@captype{table}\centering\tabletextsize\fi\fi}
\def\endfiginsert{\egroup\insert\figinsert{\vskip24pt\vbox{\unvbox0}\vskip24pt}%
\ignorespaces}

\def\endspfiginsert{\egroup\insert\figinsert{\newpage\vbox to\textheight{\vss
\unvbox0}\newpage}%
\ignorespaces}

\def\dodblfiginsert{\setbox0=\vbox\bgroup
\captionwidth=\widecaptionwidth
\hsize=\textwidth
\linewidth=\textwidth
\ifcaptypefig\def\@captype{figure}\else
\ifcaptypeplate\def\@captype{plate}\else
\def\@captype{table}\centering\fi\fi
}

\def\enddblfiginsert{\egroup\insert\figinsert{\vskip24pt\vbox{\unvbox0}\vskip24pt}}

\long\def\saveherecaption#1{\gdef\thesavedcaption{\savecaption{#1}}}

\long\def\saveherecontinuedcaption{\gdef\thesavedcaption{\continuedcaption}}

\long\def\savehereletteredcaption#1#2{\gdef\thesavedcaption{%
\gdef\theletter{#1}\savecaption{#2}}}

\long\def\savehereletteredcontinuedcaption#1{\gdef\thesavedcaption{%
\letteredcontinuedcaption{#1}}}

\long\def\saveherelabel#1{\gdef\thesavedlabel{#1}}
\saveherelabel{\relax}

\def\thesavedcaption{\relax}

\newbox\herebox
%% either figure [h] or table [h]
\newif\ifhere
\def\doonecolfighere{\vskip1sp\bgroup%
\heretrue%
\ifcaptypefig\def\@captype{figure}\else%
\ifcaptypeplate\def\@captype{plate}\else%
\def\@captype{table}\centering\fi\fi%
\setbox\herebox\vbox\bgroup%
\let\label\saveherelabel%
\let\caption\saveherecaption%
\let\letteredcaption\savehereletteredcaption%
\let\continuedcaption\saveherecontinuedcaption%
\let\letteredcontinuedcaption\savehereletteredcontinuedcaption%
}

\def\xrelax{\relax}

\def\endonecolfighere{%
\egroup%
\vskip\intextfloatskip%
\vtop{\vskip-5.5pt
%\ifappendon
%\let\thefigure\appthefigure
%\let\thetable\appthetable
%\let\theplate\apptheplate
%\let\savecaption\appcaption
%\fi
%
\ifx\thesavedcaption\xrelax
  \unvbox\herebox
\else
  \ifcaptypefig
  \unvbox\herebox
  \thesavedcaption
\else
  \ifcaptypeplate
  \unvbox\herebox
  \thesavedcaption
%
  \else
%
  \thesavedcaption
  \unvbox\herebox
  \fi
\fi\fi
%
\gdef\thesavedcaption{\relax}
\ifx\thesavedlabel\xrelax\else%
   \ifcaptypefig\else\edef\@currentlabel{\thetable}\fi
  \savelabel{\thesavedlabel}
  \gdef\thesavedlabel{\relax}
\fi
\gdef\theletter{\relax}
}\vskip1sp
\egroup
\vskip\intextfloatskip
}

\newbox\endfigbox

\def\doerr{%
\typeout{\space\space\space\space\space\space\space\space\space}
\typeout{Sorry!
Your choices following \string\figure\space or \string\table\space^^J%
are only [h] for `here', [t] for `top', [b] for `bottom', or [p]
for `page'.^^J%
I have ignored the [\one] and inserted the figure or table at this
point^^J%
on the page. See documentation if you need more help.}
\typeout{\space\space\space\space\space\space\space\space\space}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Inserts spanning both columns:

\newbox\spanbox

\def\centertabular{\bgroup\centering\savetabular}
\def\endcentertabular{\saveendtabular\vskip1pt\egroup}

%%%

\newbox\dbltopins
\newbox\dblbotins

\newskip\dblbotskip
\dblbotskip=12pt plus 3pt

\def\twocolfig{\global\captypefigtrue
\global\captypeplatefalse
\def\@captype{figure}
\@ifnextchar[\twolookforposition{\twolookforposition[t]}}%

\def\twocolplate{\global\captypeplatetrue
\global\captypefigfalse
\def\@captype{plate}
\@ifnextchar[\twolookforposition{\twolookforposition[t]}}%

\def\twocoltable{\vskip1pt\tabletextsize
\global\captypefigfalse
\global\captypeplatefalse
\def\@captype{table}
\@ifnextchar[\twolookforposition{\twolookforposition[t]}}%

\newcount\dblfigandtabnumber
\dblfigandtabnumber=1
\newbox\dblfigandtabbox

\def\starttwocolfloat{\ifvmode\else\unskip\fi\expandafter%
  \ifx\csname dblfigandtabbox\the\dblfigandtabnumber\endcsname\relax%
  \expandafter%
  \newbox\csname dblfigandtabbox\the\dblfigandtabnumber\endcsname%
  \fi%
\global\setbox\csname dblfigandtabbox\the\dblfigandtabnumber\endcsname=
\vbox\bgroup\let\label\dblcatchlabel%
%
  \ifcaptypefig\def\@captype{figure}\else%
    \ifcaptypeplate\def\@captype{plate}\else%
    \def\@captype{table}\centering\tabletextsize%
    \fi%
  \fi%
\captionwidth=\widecaptionwidth
\hsize=\textwidth
\linewidth=\textwidth\relax
%
\let\caption\dblcatchcaption
\let\letteredcaption\dblcatchletteredcaption
\let\continuedcaption\dblcatchcontinuedcaption
\let\letteredcontinuedcaption\dblcatchletteredcontinuedcaption
\let\label\dblcatchlabel
}

\newcount\dblloopnum
\newcount\savedblfigandtabnumber
%\newcount\savefigandtabnumber

\def\xdbltopfloat{dbltopfloat}
\def\xdblbotfloat{dblbotfloat}

\def\endtwocolfloat{%
%\ifcaptypefig\else%
%\ifcaptypeplate\else%
%\ifnobottomtabline\global\nobottomtablinefalse\vskip-6.5pt\else%
%\vskip2pt
%\thetablelines\fi\fi\fi%
%
\egroup%
\ifx\dbltoporbotfloat\xdbltopfloat%
%%++
\ifcaptypefig%
\expandafter\gdef\csname dbltopfloat\the\dblfigandtabnumber\endcsname{%
\vbox{\vskip\PushTwoColTopFig%
\copy\csname dblfigandtabbox\the\dblloopnum\endcsname%
  \vskip\captionskip%
  \csname dblcaption\the\dblloopnum\endcsname%
  \csname dblletteredcaption\the\dblloopnum\endcsname%
  \csname dblcontinuedcaption\the\dblloopnum\endcsname%
  \csname dblletteredcontcaption\the\dblloopnum\endcsname%
\ifredefining%
\xdef\@currentlabel{\thefigure}%
\csname dbllabel\the\dblloopnum\endcsname\fi}%
  \vskip\intextfloatskip %%
  \vskip-5pt% because there is a space above the top of text, from topskip?
\ifredefining%
\expandafter\gdef\csname dbltopfloat\the\dblloopnum\endcsname{}\fi}%
\else\ifcaptypeplate%
\expandafter\gdef\csname dbltopfloat\the\dblfigandtabnumber\endcsname{%
\vbox{\vskip\PushTwoColTopFig%
\copy\csname dblfigandtabbox\the\dblloopnum\endcsname%
  \vskip\captionskip%
  \csname dblcaption\the\dblloopnum\endcsname%
  \csname dblletteredcaption\the\dblloopnum\endcsname%
  \csname dblcontinuedcaption\the\dblloopnum\endcsname%
  \csname dblletteredcontcaption\the\dblloopnum\endcsname%
\ifredefining%
\xdef\@currentlabel{\theplate}%
\csname dbllabel\the\dblloopnum\endcsname\fi}%
  \vskip\intextfloatskip %%
  \vskip-5pt% because there is a space above the top of text, from topskip?
\ifredefining%
\expandafter\gdef\csname dbltopfloat\the\dblloopnum\endcsname{}\fi}%
\else% table
\expandafter\gdef\csname dbltopfloat\the\dblfigandtabnumber\endcsname{%
\vbox{\vskip\PushTwoColTopTab%
  \csname dblcaption\the\dblloopnum\endcsname%
  \csname dblletteredcaption\the\dblloopnum\endcsname%
  \csname dblcontinuedcaption\the\dblloopnum\endcsname%
  \csname dblletteredcontcaption\the\dblloopnum\endcsname%
  \vskip\captionskip%
  \copy\csname dblfigandtabbox\the\dblloopnum\endcsname}%
  \vskip\intextfloatskip %%
  \vskip-5pt% because there is a space above the top of text, from topskip?
\ifredefining%
\csname dbllabel\the\dblloopnum\endcsname%
\expandafter\gdef\csname dbltopfloat\the\dblloopnum\endcsname{}\fi}%
%
\fi\fi%
%
\else% bottom float
%
\ifcaptypefig%
\expandafter\gdef\csname dblbotfloat\the\dblfigandtabnumber\endcsname{%
  \vskip\intextfloatskip%
\vbox{\copy\csname dblfigandtabbox\the\dblloopnum\endcsname%
  \vskip\captionskip%
  \csname dblcaption\the\dblloopnum\endcsname%
  \csname dblletteredcaption\the\dblloopnum\endcsname%
  \csname dblcontinuedcaption\the\dblloopnum\endcsname%
  \csname dblletteredcontcaption\the\dblloopnum\endcsname%
\ifredefining%
\csname dbllabel\the\dblloopnum\endcsname\fi%
\vskip\PushTwoColBotFig}%???
 \ifredefining%
\expandafter\gdef\csname dblbotfloat\the\dblloopnum\endcsname{}\fi}%
\else%
%%
\ifcaptypeplate%
\expandafter\gdef\csname dblbotfloat\the\dblfigandtabnumber\endcsname{%
  \vskip\intextfloatskip%
\vbox{\copy\csname dblfigandtabbox\the\dblloopnum\endcsname%
  \vskip\captionskip%
  \csname dblcaption\the\dblloopnum\endcsname%
  \csname dblletteredcaption\the\dblloopnum\endcsname%
  \csname dblcontinuedcaption\the\dblloopnum\endcsname%
  \csname dblletteredcontcaption\the\dblloopnum\endcsname%
\ifredefining%
\csname dbllabel\the\dblloopnum\endcsname\fi%
\vskip\PushTwoColBotFig}%
 \ifredefining%
\expandafter\gdef\csname dblbotfloat\the\dblloopnum\endcsname{}\fi
}%
\else% table
\expandafter\gdef\csname dblbotfloat\the\dblfigandtabnumber\endcsname{%
  \vskip\intextfloatskip
\vbox{%
  \csname dblcaption\the\dblloopnum\endcsname%
  \csname dblletteredcaption\the\dblloopnum\endcsname%
  \csname dblcontinuedcaption\the\dblloopnum\endcsname%
  \csname dblletteredcontcaption\the\dblloopnum\endcsname%
  \vskip\captionskip
\ifredefining%
\csname dbllabel\the\dblloopnum\endcsname\fi%
  \copy\csname dblfigandtabbox\the\dblloopnum\endcsname%
\vskip\PushTwoColBotTab}%
\ifredefining%
\expandafter\gdef\csname dblbotfloat\the\dblloopnum\endcsname{}\fi}%
%
\fi\fi\fi%
%%
\global\advance\dblfigandtabnumber by1 \relax}

\newbox\dblspanherebox

\def\dodblfigurehere{\global\setbox\dblspanherebox=\vbox\bgroup
\let\label\saveherelabel
\ifcaptypefig\def\@captype{figure}\else
\ifcaptypeplate\def\@captype{plate}\else
\def\@captype{table}\centering\fi\fi
\vskip\intextfloatskip
\captionwidth=\widecaptionwidth
\hsize=\textwidth
\linewidth=\textwidth
\let\label\saveherelabel
\let\caption\saveherecaption
\let\letteredcaption\savehereletteredcaption
\let\continuedcaption\saveherecontinuedcaption
\let\letteredcontinuedcaption\savehereletteredcontinuedcaption}

\def\enddodblfigurehere{\vskip\intextfloatskip\egroup%
\aftergroup\endcolsneatly}


%%
\def\endcolsneatly{\ifdim\lastskip=2sp
\vskip-9.5pt\null\fi
\endtwocolumns
\vbox{\vskip\intextfloatskip
\captionwidth=\widecaptionwidth
\ifcaptypefig\def\@captype{figure}\else
\ifcaptypeplate\def\@captype{plate}\else
\def\@captype{table}\centering\fi\fi
%\ifappendon
%\let\thefigure\appthefigure
%\let\thetable\appthetable
%\let\theplate\apptheplate
%\let\savecaption\appcaption
%\fi
%
\ifx\thesavedcaption\xrelax
\unvbox\dblspanherebox
\vskip\intextfloatskip
\else
  \ifcaptypefig
\unvbox\dblspanherebox
  \thesavedcaption
\vskip\intextfloatskip
%
  \else
\ifcaptypeplate
\unvbox\dblspanherebox
  \thesavedcaption
\vskip\intextfloatskip
%
\else
\vskip\intextfloatskip
  \thesavedcaption
\unvbox\dblspanherebox
\vskip\intextfloatskip
  \fi
\fi\fi
%
\gdef\thesavedcaption{\relax}
\ifx\thesavedlabel\xrelax\else\savelabel{\thesavedlabel}
\gdef\thesavedlabel{\relax}\fi
\gdef\theletter{\relax}
\vskip\intextfloatskip}
\twocolumns\null
\vskip2sp}

\def\twolookforposition[#1]{\defone#1*%
\let\go\starttwocolfloat%
\ifgalley%
  \if\one p \def\one{s}%
  \else%
  \def\one{i}%
  \fi%
\fi%
\ifjdraft%
\expandafter\ifx\csname setkeys\endcsname\relax\else
\setkeys{Gin}{draft=false}\fi
  \if\one p \def\one{s}%
  \else%
  \def\one{i}%
  \fi%
\fi%
%\ifappendon\def\one{h}\fi%
%%
\if\one h%
  \ifcaptypeplate%
  \else%
    \ifcaptypefig%
    \dofigmessage%
    \else%
    \dotabmessage%
    \fi%
  \fi%
    \ifgalley%
     \gdef\dbltoporbotfloat{dbltopfloat}%
     \else% not galley ==>
     \let\go\dodblfigurehere%
    \expandafter\gdef\csname endfigure*\endcsname{\enddodblfigurehere}%
    \expandafter\gdef\csname endtable*\endcsname{\enddodblfigurehere}%
    \expandafter\gdef\csname endplate*\endcsname{\enddodblfigurehere}%
    \fi% end ifgalley
\else%
   \if\one t%
   \gdef\dbltoporbotfloat{dbltopfloat}%
   \else%
      \if\one b%
      \gdef\dbltoporbotfloat{dblbotfloat}%
      \else%
         \if\one p%
         \let\go\startpagefloat%
    \expandafter\gdef\csname endfigure*\endcsname{\endpagefloat}
    \expandafter\gdef\csname endtable*\endcsname{\endpagefloat}
    \expandafter\gdef\csname endplate*\endcsname{\endpagefloat}
         \else%
            \if\one i% for galley mode
            \let\go\dodblfiginsert%
    \expandafter\gdef\csname endfigure*\endcsname{\enddblfiginsert}
    \expandafter\gdef\csname endtable*\endcsname{\enddblfiginsert}
    \expandafter\gdef\csname endplate*\endcsname{\enddblfiginsert}
            \else%
               \if\one s% for special galley mode, for [p]
               \let\go\dodblfiginsert%
    \expandafter\gdef\csname endfigure*\endcsname{\endspfiginsert}%
    \expandafter\gdef\csname endtable*\endcsname{\endspfiginsert}
    \expandafter\gdef\csname endplate*\endcsname{\endspfiginsert}
               \else%
\doerr%
              \fi%
           \fi%
        \fi%
     \fi%
   \fi%
\fi\go}%

\let\saveenddocument\enddocument

\def\enddocument{%
\ifdim\ht\dbltopins>0pt\vbox{\unvbox\dbltopins}\fi
\ifdim\ht\dblbotins>0pt\vbox{\unvbox\dblbotins}\fi
\ifnum\pagefloatnumber>\pageloopnum
\newpage
\insertpage{\hsize=\textwidth
\linewidth=\textwidth
\csname pagefloat\the\pageloopnum \endcsname}\fi%
\immediate\write\@auxout{\string\gdef\string\lastpage{\thepage}}
\saveenddocument}

\splittopskip=\topskip

\def~{\penalty\@M{ }}

\def\mathleftline{\vskip-\parskip
\hbox to\textwidth{\hrulefill\hskip.5\textwidth}\vskip-\abovedisplayskip}

\def\mathrightline{\hbox to\textwidth{\hskip.5\textwidth\hrulefill}\vskip18pt}

\let\topline\mathleftline
\let\botline\mathrightline

%% End Two Column Macros
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\let\savefigure\figure
\let\saveendfigure\endfigure


%%%Packages installing
\RequirePackage{graphicx}
\RequirePackage{epstopdf}
\RequirePackage{amssymb}
\RequirePackage{amsfonts}
\RequirePackage{amsmath}



\RequirePackage[usenames]{xcolor}
\definecolor{signitext}{cmyk}{1,.45,0,.18}
\definecolor{signiback}{cmyk}{.15,.07,0,.03}
\gdef\significancetext#1{\gdef\@significancetext{#1}}
\let\@significancetext\@empty

\def\article{\twocolumns
\bgroup
\setlength{\fboxsep}{6pt}
\ifx \@significancetext\@empty
\renewcommand\thefootnote{}%
\footnotetext{\vbox to 3.7cm{\significanceheadfont Reserved for Publication Footnotes\vfill}}%
\else
\renewcommand\thefootnote{}%
\footnotetext{\vbox{\fcolorbox{signitext}{signiback}{\parbox{232pt}{\par\vspace*{2pt}{\significanceheadfont \color{signitext}Significance}\par\vspace*{12pt}%
                                                    {\color{signitext}\significancefont\@significancetext}}}\par\vspace*{4.5pt}
                				   {\vbox to 3.7cm{\significanceheadfont Reserved for Publication Footnotes\vfill}}}}
\fi
\egroup
}



\def\endarticle{\endtwocolumns\newpage
\expandafter\gdef\csname table*\endcsname{\@dblfloat{table}}
\expandafter\gdef\csname endtable*\endcsname{\end@dblfloat}
\expandafter\gdef\csname figure*\endcsname{\@dblfloat{figure}}
\expandafter\gdef\csname endfigure*\endcsname{\end@dblfloat}
\gdef\figure{\futurelet\next\lookforbrac}}

\def\lookforbrac{\if\next[
\let\go\figone\else \let\go\figtwo\fi\go}

\def\figone[#1]{\@float{figure}[#1]\hsize276pt}
\def\figtwo{\@float{figure}\hsize276pt}


\newif\ifjdraft
\newdimen\captionwidth
\newdimen\widecaptionwidth
\newif\ifgalley
\let\tabletextsize\normalsize




\def\inlinefrac#1#2{%
   \mbox{\raisebox{.8mm}{%
       $\textstyle #1$%
                        }
      $\!\!\! / \!\!\!$
      \raisebox{-.8mm}{$\textstyle #2$}}%
                       }

\newcount\tablenoteloopnum
\newcount\tablenotenum

\def\tablenote#1{\global\advance\tablenotenum by 1\relax
$^{\@fnsymbol{\the\tablenotenum}}$\expandafter\gdef\csname
tabnote\the\tablenotenum\endcsname{#1}}

\def\tablenotes{\tablenoteloopnum=\tablenotenum
\global\advance\tablenoteloopnum by 1
\tablenotenum=0
{\figtextfont
\leftskip=0pt \rightskip=\leftskip
\parfillskip=0pt plus 1 fil
\loop
\vskip2pt
\noindent
\global\advance\tablenotenum by 1
\ifnum\tablenotenum<\tablenoteloopnum
$^{\@fnsymbol{\the\tablenotenum}}$\csname
tabnote\the\tablenotenum\endcsname
\repeat}
}

%% convenient macros from author:


\def\o{\omega}
\def\pr{\partial}
\def\d{\delta}
\def\a{\alpha}
\def\p{\phi}
\def\dst{\displaystyle}
\def\chit{\tilde{\chi}}
\def\etat{\tilde{\eta}}

\def\fdb{\mathbb{R}/_{\dst{ \mathbb{Z}}}}
\def\fd{\mathbb{R}/_{\scriptstyle \mathbb{Z}}}
\def\f12{\frac{1}{2}}
\def\fr32{\frac{3}{2}}
\def\vs{\vspace{10pt}}
\def\vsm{\vspace{-10pt}}
\def\xb{\bar{x}}
\def\yb{\bar{y}}
\def\tb{\bar{t}}
\def\xt{\tilde{x}}
\def\yt{\tilde{y}}
\def\gtt{\tilde{g}}
\def\vptt{\tilde{\varphi}}
\def\htt{\tilde{h}}
\def\ktt{\tilde{k}}
\def\ub{\bar{u}}
\def\hb{\bar{h}}
\def\vep{\varepsilon}
\def\vs{\vspace{10pt}}
\def\be{\begin{equation}}
\def\ee{\end{equation}}
\newcommand{\mfrac}[2]{\frac{\displaystyle{#1}}{\displaystyle{#2}}}


\newtheorem{theorem}{Theorem}
\newtheorem{lemma}{Lemma}
\newtheorem{corollary}[theorem]{Corollary}
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{definition}{Def\,inition}
\newtheorem{claim}{Claim}
\newtheorem{conjecture}[theorem]{Conjecture}
\newtheorem{observation}[theorem]{Observation}
\newtheorem{problem}[theorem]{Problem}
\newtheorem{notation}[theorem]{Notation}
\newtheorem{alemma}{Lemma}
\newtheorem{remark}{Remark}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


\def\acknowledgments{\vskip14pt\bgroup\footnotesize\baselineskip 8pt
\materialfont
\def\textit##1{{\materialitfont ##1}}
\def\it{\materialitfont}
\def\bf{\materialbffont}
\def\textbf##1{{\materialbffont ##1}}
\noindent{\bf ACKNOWLEDGMENTS.} \relax\ignorespaces}
\def\endacknowledgments{\vskip1sp\egroup}

%% Just in case author enters any of the below, it will still work correctly
\let\acknowledgment\acknowledgements
\let\endacknowledgment\endacknowledgments

\let\acknowledgement\acknowledgements
\let\endacknowledgement\endacknowledgments

\let\acknowledgements\acknowledgements
\let\endacknowledgements\endacknowledgments

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% April 25, 2006
\let\dospecialaccents\relax
\def\blackbox#1{}

% turn off side squares
\def\dosidesquare#1#2#3{}
\let\resetsidesquares\relax
\let\makesidesquares\relax
\newif\iffirsttime

\def\thanks#1{\footnotemark \protected@xdef \@thanks
{\@thanks \protect \footnotetext [\the \c@footnote ]{\hsize=247pt\relax #1}}}

%% Keywords
{\global\let\saveline|
\gdef\linewithspace{\hskip3pt$\saveline$\hskip6pt}
\catcode`|=\active
\gdef\keywords{\bgroup\catcode`|=\active\xkeywords}
\gdef\xkeywords#1{\vskip1sp\noindent{\let|=\linewithspace
\affilfont#1}\vskip11pt\egroup}
}

\def\abbreviations#1{\vskip1sp\noindent{\affilfont Abbreviations: #1}\vskip12pt}

%% Authors can change this dimension to move the
%% printed page up or down on the physical page. Making
%% the negative dimension larger will move it up on the
%% page, smaller will move it down.

\voffset-.75in

%%% Default copyright year
\copyrightyear{2007}

\def\conflictofinterest#1{{\let\@thefnmark\relax\@footnotetext{\vrule height8pt width0pt\relax\hsize=247pt
\relax\footnotesize\footnotefont #1\vrule depth 6pt width0pt}}}

\def\footcomment#1{{\let\@thefnmark\relax\@footnotetext{\vrule height8pt width0pt\relax\hsize=247pt\relax\footnotesize\footnotefont #1\vrule depth 6pt width0pt}}}

\def\track#1{\footcomment{#1}}

\url{www.pnas.org --- ---}
\issuedate{Issue Date}
\copyrightyear{2007}
\volume{Volume}
\issuenumber{Issue Number}
%\setcounter{page}{2687}
\footlineauthor{Footline Author}

\long\def\@footnotetext#1{\insert\footins{%
    \reset@font\footnotesize
    \interlinepenalty\interfootnotelinepenalty
    \splittopskip\footnotesep
    \splitmaxdepth \dp\strutbox \floatingpenalty \@MM
    \hsize\columnwidth \@parboxrestore
    \protected@edef\@currentlabel{%
       \csname p@footnote\endcsname\@thefnmark
    }%
    \color@begingroup
      \@makefntext{%
        \rule\z@\footnotesep\ignorespaces\footnotefont
\hsize=247pt
#1\@finalstrut\strutbox}%
    \color@endgroup}}%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Special Use Fonts
%%

%% Computer Modern font calls. These will be redefined
%% if PostScript fonts are used, \usepackage{pnastwof}

%% Times-Roman
\def\timesroman{cmr10 }
\def\timesbold{cmbx10 }
\def\timesitalic{cmti10 }
\def\timesbolditalic{cmbx10 }

%% Helvetica
\def\helvetica{cmss10 }
\def\helveticabold{cmssbx10 }
\def\helveticaboldoblique{cmssbx10 }
\def\helveticaoblique{cmssi10 }

\def\frutiger{cmss10 }
\def\frutigermed{cmss10 }
\def\frutigerbold{cmssbx10 }
\def\frutigerboldoblique{cmssi10 }
\def\frutigeroblique{cmssi10 }

\let\frutigerboldcondensed\helveticabold

%% Special Use Fonts ==>>


\font\footnotefont\helvetica at 6pt

\font\dropcapfont=\frutigerbold at 26pt
\font\sideboxfont= \helvetica at 6pt

%% Running feet
  \font\footlinefont=\frutiger at6.5pt%7pt
  \font\footlineit=\helveticaoblique at6.5pt%7pt
  \font\foliofont=\frutigerbold at6.5pt%7pt

\font\titlefont=\frutigerboldcondensed at 22pt % 24pt baseline
  \font\authorfont=\frutigerbold at 9pt % 11 pt baseline
  \font\affilfont=\frutiger at 7pt % 8pt baseline


  \font\copyrightfont=\frutiger at 6pt
  \font\copyrighttextfont=\frutiger at 6pt

%% Abstract fonts
\font\xabstractfont=\frutigerbold at 8pt
\def\abstractfont{\sf\xabstractfont\dospecialaccents}
\font\abstractit=\helveticaboldoblique at 8pt

%% Section fonts
\font\sectionfont=\frutigerbold at 10pt
\font\subsectionfont=\frutigerbold at9pt

\font\significanceheadfont=\frutigerbold at 9pt
\font\significancefont=\frutigerbold at 8pt


%% Theorem fonts
  \font\theoremfont=\frutigerbold at 8pt

%ok
%% Caption
  \font\fignumfont=\frutigerbold at 9pt
\font\xfigtextfont=\frutiger at 7pt
  \def\figtextfont{\small\dospecialaccents\xfigtextfont}

%% Table font
\font\tabletextfont=\frutiger at 9pt
\font\tabtextfont\frutigerbold at 9pt
\font\tabnumfont\frutigerbold at 9pt

%% Materials and Methods Fonts
\font\materialfont=\frutiger at 7pt
\font\materialbffont=\helveticabold at 7pt
\font\materialitfont=\helveticaboldoblique at 7pt

%% Bibliography fonts
\font\xbibfont=\frutigerbold at 6pt
\def\bibfont{\sf\xbibfont\dospecialaccents}
\font\bibitfont\helveticaboldoblique at 6pt
\font\bibbffont\helveticabold at 6pt

%% dec 4, 2006
\def\@tabular{\global\tablenotenum=0\relax
\leavevmode \hbox \bgroup $\let\@acol\@tabacol
   \let\@classz\@tabclassz
   \let\@classiv\@tabclassiv \let\\\@tabularcr\@tabarray}

%% dec 4, 2006
\def\@tabular{\global\tablenotenum=0\relax
\leavevmode \hbox \bgroup $\let\@acol\@tabacol
   \let\@classz\@tabclassz
   \let\@classiv\@tabclassiv \let\\\@tabularcr\@tabarray}

%%%%%%% Jan 2008 Changes:
   \def\materials{\vskip6pt\bgroup
\parskip=8pt
\baselineskip=9pt
\materialfont
\def\textit##1{{\materialitfont ##1}}
\def\it{\materialitfont}
\def\bf{\materialbffont}
\def\textbf##1{{\materialbffont ##1}}
\def\section##1{\noindent{\materialbffont ##1. }}
\def\subsection##1{\noindent{\materialbffont ##1. }}
\noindent {\subsectionfont Materials and Methods}\\[2pt]
}
\def\endmaterials{\vskip1sp\egroup}


\def\appendix{\originalappendix
\vskip12pt\noindent{\subsectionfont Appendix}\@ifnextchar[{\xappendix}{\yappendix}}

\def\xappendix[#1]{\subsectionfont:\hskip10pt\relax #1\vskip2pt
\normalsize\noindent\ignorespaces}
\def\yappendix{\vskip2pt\normalsize\noindent\ignorespaces}

\widowpenalty=10000
\clubpenalty=10000

\newbox\tablewide
\def\endtabular{\crcr\egroup\egroup $\egroup\egroup
\centerline{\vbox{\hsize\wd\tablewide
\currtabcaption\vskip1pt
}}
\dimen0=\wd\tablewide
\centerline{\hbox{\unhbox\tablewide}}
\centerline{\vtop{\hsize=\dimen0 \tablenotes}}
\global\let\currtabcaption\relax}

\expandafter \let \csname endtabular*\endcsname = \endtabular

\def\tabular{\global\setbox\tablewide\hbox\bgroup
\let\@halignto\@empty\@tabular}

\@namedef{tabular*}#1{%
\global\setbox\tablewide\hbox\bgroup
 \setlength\dimen@{#1}%
   \edef\@halignto{to\the\dimen@}\@tabular}

\newcommand\originalappendix{\par
  \setcounter{section}{0}%
  \setcounter{subsection}{0}%
  \gdef\thesection{\@Alph\c@section}}

\let\currtabcaption\relax


\def\footcomment#1{\typeout{^^J^^J The \string\footcomment{}\space command is
No Longer Used^^J^^J}}

\def\track#1{\typeout{^^J^^J The \string\track{}\space command is
No Longer Used^^J^^J}}

\def\conflictofinterest#1{\typeout{^^J^^J The \string\conflictofinterest{}\space command is
No Longer Used^^J^^J}}

%% nothing will happen, so PNAS can get info, but nothing will be printed
\def\nothanks#1{}

\let\thanks\nothanks


% substitute for \footcomment:

\def\abbreviations#1{\vskip1sp\noindent{\affilfont Abbreviations: #1}
\vskip12pt}
\endinput
Change Log:
October 18, 2006
Changed the counter used in loop in maketitle to
fix bug that prevented all affiliations from being
listed; added $\saveline$ to make line work right
in keywords

dec 4, 2006: fixed bug in table notes, by
resetting tablenotenum every time tabular
is used

%\def\currversion{October 18, 2006}
%\def\currversion{July 10, 2006}
%\def\currversion{May 1, 2006}
%\def\currversion{August 27, 2004}