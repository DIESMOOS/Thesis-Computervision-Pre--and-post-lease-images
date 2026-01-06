%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% mscthesis.cls
% Thesis document definition adapted from multiple 
% sources (ACM, ETH DISCO thesis) for the use of Univesity of Amsterdam
% Information Science and Data Science students
% Adapted by Riccardo Fiorista (riccardo.fiorista@proton.me)

% As you're asking, yes this is all very similar to the mscthesis.cls however,
% it treats line-numbering differently. There was an issue with the title page
% and I couldn't resolve it in the first place.
% Duplicating a bit of code seemed the most straight forward action to take.
% I hope you can forgive me Robert C. Martins.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{thesisdesign}
              [2022/08/31 Modified from the ACM paper definition by Riccardo Fiorista]

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LOAD PACKAGES AND DEFINE ADDITIONAL COMMANDS
% Modified from ETH Zurich DISCO Thesis Template
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\AtEndOfClass{
% \RequirePackage{titling}

\RequirePackage{amsmath}
\RequirePackage{amsthm}
% Clashes with the updated package in the ACM class
% \RequirePackage{amssymb}
\RequirePackage{setspace}
\RequirePackage{booktabs}
\RequirePackage[title]{appendix}

% GANNT Charts
\RequirePackage{pgfgantt}


% Define mathematical commands
%% Define variable representing a matrix
\usepackage{bm}
\newcommand{\matvar}[1]{\bm{\mathit{#1}}}

%% Define math-styled operator for argmax optimization
\DeclareMathOperator*{\argmax}{argmax} % thin space, limits underneath in displays

%% Theorem, lemmas, and more ...
\theoremstyle{plain}
\newtheorem{theorem}{Theorem}[section]
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{corollary}[theorem]{Corollary}
\newtheorem{observation}[theorem]{Observation}

\theoremstyle{definition}
\newtheorem{definition}[theorem]{Definition}
\newtheorem{problem}[theorem]{Problem}
\newtheorem{assumption}[theorem]{Assumption}
\newtheorem{example}[theorem]{Example}

\theoremstyle{remark}
\newtheorem{claim}[theorem]{Claim}
\newtheorem{remark}[theorem]{Remark}

% Enables the use of colours
\RequirePackage{xcolor}
\definecolor{darkblue}{rgb}{0,0,.5}

% Enables clickable links in the PDF and additional PDF specific configuration options.
\PassOptionsToPackage{
    colorlinks=true,
    linkcolor=darkblue, 
    urlcolor=darkblue, 
    citecolor=darkblue,
    raiselinks=true,
    bookmarks=true,
    bookmarksopenlevel=1,
    bookmarksopen=true,
    bookmarksnumbered=true,
    hyperindex=true,
    plainpages=false,
    pdfpagelabels=true,
    pdfstartview=FitH,
    pdfstartpage=1,
    pdfpagelayout=OneColumn
}{hyperref}

% Allows to use nice todo notes on the paper margin
\RequirePackage{todonotes}

% Allows line numbering
\RequirePackage{lineno}
\iflinenumbering
    \linenumbers
\fi

\RequirePackage{flushend}
\RequirePackage{balance}

% Additional packages
\newcommand\xst{\textsuperscript{st}}
\newcommand\xnd{\textsuperscript{nd}}
\newcommand\xrd{\textsuperscript{rd}}
\newcommand\xth{\textsuperscript{th}}

\usepackage[english]{babel}
\usepackage{chronology}
\usepackage{graphicx}
\usepackage{subcaption}
\usepackage{xspace}
\usepackage{tabularx}

% Code listings for pseudo code
\usepackage{algorithm2e}
\SetKwComment{Comment}{/* }{ */}
\RestyleAlgo{ruled}
\usepackage{algpseudocode}

\usepackage{listings}
\definecolor{codegreen}{rgb}{0,0.6,0}
\definecolor{codegray}{rgb}{0.5,0.5,0.5}
\definecolor{codepurple}{rgb}{0.58,0,0.82}
\definecolor{backcolour}{rgb}{0.95,0.95,0.92}
\lstdefinestyle{mystyle}{
    backgroundcolor=\color{backcolour},   
    commentstyle=\color{codegreen},
    keywordstyle=\color{magenta},
    numberstyle=\tiny\color{codegray},
    stringstyle=\color{codepurple},
    basicstyle=\ttfamily\footnotesize,
    breakatwhitespace=false,         
    breaklines=true,                 
    captionpos=b,                    
    keepspaces=true,                 
    numbers=left,                    
    numbersep=5pt,                  
    showspaces=false,                
    showstringspaces=false,
    showtabs=false,                  
    tabsize=2
}
\lstset{style=mystyle}

\addto\extrasenglish{
    \def\sectionautorefname{Section}
    \def\subsectionautorefname{Section}
    \def\subsubsectionautorefname{Section}
}

% Defining todo commands
\setlength{\marginparsep}{\dimexpr\marginparsep + 2mm\relax}
\ifdraftmargins
    % As suggested in https://tex.stackexchange.com/a/404766
    \setlength{\paperwidth}{\dimexpr\paperwidth + 6cm\relax}
    \setlength{\oddsidemargin}{\dimexpr\oddsidemargin + 3cm\relax}
    \setlength{\evensidemargin}{\dimexpr\evensidemargin + 3cm\relax}
    \setlength{\marginparwidth}{\dimexpr\marginparwidth + 3cm\relax}

    \newcommand{\TODO}[1]
    {\todo[prepend, caption={\textbf{Todo}}]{\renewcommand{\baselinestretch}{0.4} \selectfont#1\par}}
    \newcommand{\REMARK}[1]
    {\todo[prepend, color=green, caption={\textbf{Remark}}]{\renewcommand{\baselinestretch}{0.85} \selectfont#1\par}}
\else
    \setlength{\marginparwidth}{1.2cm}
    \newcommand{\TODO}[1]
    {\todo[prepend, caption={\textbf{Todo}}, size=\footnotesize]{\renewcommand{\baselinestretch}{0.4} \selectfont#1\par}}
    \newcommand{\REMARK}[1]
    {\todo[prepend, color=green, caption={\textbf{Remark}}, size=\footnotesize]{\renewcommand{\baselinestretch}{0.85} \selectfont#1\par}}
\fi

% Suppresses ACM-specific content
\settopmatter{printacmref=false}
\renewcommand\footnotetextcopyrightpermission[1]{} 
}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DECLARE DATA COMMANDS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Thesis data
% `title` and `keywords` are already defined
\newcommand\thesiskeywords[1]{\keywords{#1}\gdef\@thesiskeywords{#1}}
\newcommand\thesistitle[1]{\title{#1}\gdef\@thesistitle{#1}}
\newcommand\thesissubtitle[1]{\subtitle{#1}\gdef\@thesissubtitle{#1}}
\newcommand\thesisdate[1]{\date{#1}\gdef\@thesisdate{#1}}

% Titlepage data

% Student data
\newcommand\authorname[1]{\author{#1}\gdef\@authorname{#1}}
\newcommand\authoremail[1]{\email{#1}\gdef\@authoremail{\href{mailto:#1}{#1}}}
\newcommand\authorid[1]{\gdef\@authorid{#1}}

% Supervisor data
\newcommand\uvasupervisorname[1]{\gdef\@firstsupervisorname{#1}}
\newcommand\uvasupervisoraffiliation[1]{\gdef\@firstsupervisoraffiliation{#1}}
\newcommand\uvasupervisoremail[1]{\gdef\@firstsupervisoremail{\href{mailto:#1}{#1}}}

\newcommand\externalsupervisorname[1]{\gdef\@externalsupervisorname{#1}}
\newcommand\externalsupervisoraffiliation[1]{\gdef\@externalsupervisoraffiliation{#1}}
\newcommand\externalsupervisoremail[1]{\gdef\@externalsupervisoremail{\href{mailto:#1}{#1}}}

\newcommand\examinername[1]{\gdef\@examinername{#1}}
\newcommand\examineraffiliation[1]{\gdef\@examineraffiliation{#1}}
\newcommand\examineremail[1]{\gdef\@examineremail{\href{mailto:#1}{#1}}}

% Institutional data
\def\thesisdegree{Master Information Studies}
\def\thesisfaculty{Faculty of Science}
\def\thesisinstitution{University of Amsterdam}
\def\thesiscity{Amsterdam}
\def\thesiscountry{The Netherlands}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DECLARE HELPER VARIABLES AND COMMANDS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newif\iflinenumbering
\linenumberingtrue

\newif\ifdraftmargins
\draftmarginsfalse
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DECLARE OPTIONS FOR IS AND DS STYLES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\DeclareOption{nolinenumbering}{\linenumberingfalse}

\DeclareOption{draftmargins}{
    \draftmarginstrue
}

% Catch all options that are not defined here and send them over to the base-class
\DeclareOption*{\PassOptionsToClass{\CurrentOption}[sigconf]{acmart}}

\ProcessOptions\relax
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DECLARE OPTIONS FOR IS AND DS STYLES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Patch for lineno to work with the AmsMath environments from https://latex.org/forum/viewtopic.php?t=163
\newcommand*\patchAmsMathEnvironmentForLineno[1]{%
  \expandafter\let\csname old#1\expandafter\endcsname\csname #1\endcsname
  \expandafter\let\csname oldend#1\expandafter\endcsname\csname end#1\endcsname
  \renewenvironment{#1}%
     {\linenomath\csname old#1\endcsname}%
     {\csname oldend#1\endcsname\endlinenomath}}% 
\newcommand*\patchBothAmsMathEnvironmentsForLineno[1]{%
  \patchAmsMathEnvironmentForLineno{#1}%
  \patchAmsMathEnvironmentForLineno{#1*}}%
\AtBeginDocument{%
\patchBothAmsMathEnvironmentsForLineno{equation}%
\patchBothAmsMathEnvironmentsForLineno{align}%
\patchBothAmsMathEnvironmentsForLineno{flalign}%
\patchBothAmsMathEnvironmentsForLineno{alignat}%
\patchBothAmsMathEnvironmentsForLineno{gather}%
\patchBothAmsMathEnvironmentsForLineno{multline}%
}

% Weird formatting error with lineno when linenumbering is enabled -> it adds an empty page
% after the title page. With this command we account for that
\newcommand\fixemptypage{
    \iflinenumbering
        \AtBeginShipoutNext{\AtBeginShipoutDiscard}
    \fi
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\LoadClass[sigconf, screen=true, review=false, twocolumn]{acmart}
% To allow multi-line inputs in fields such as subtitle
% Taken from Solution 2 here:
% https://9to5science.com/hyperref-warning-token-not-allowed-in-a-pdf-string
\pdfstringdefDisableCommands{%
  \def\\{}%
  \def\texttt#1{<#1>}%
}
