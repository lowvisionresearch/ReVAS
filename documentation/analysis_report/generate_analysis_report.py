'''
Generates a Latex file that juxtaposes reference frames
and eye position trace plots.
'''

import os

header = """\documentclass[11pt]{article}
\\usepackage{fullpage}
\\usepackage{amsfonts}
\\usepackage{graphicx}
\\usepackage[export]{adjustbox}
\\setlength\parindent{0pt}

\\begin{document}

\\title{Analysis Report}
\\author{Matthew Sit, Derek Wan}
\\date{September 9, 2017}
\\maketitle

"""

footer = "\end{document}"

def section(videoClassification):
    videoClassification = videoClassification.replace("_", "\_")
    return "\section{" + videoClassification + "}\n\n"

def subsection(videoClassification, filename):
    subtitle = filename[:filename.index("_dwt")]
    subtitle = subtitle.replace("_", "\_")

    plotname = filename[:filename.index("_dwt")] + ".jpg"
    paramsname = filename[:filename.index("_dwt")] + ".txt"

    file = open("params/" + videoClassification + "/" + paramsname, "r")
    params = file.read().replace("_", "\_").replace("NEWLINE", "\\\\\n").replace(":", ": ")
    params = params.replace("Coarse Parameters", "\\textbf{Coarse Parameters}")
    params = params.replace("Fine Parameters", "\\textbf{Fine Parameters}")
    params = params.replace("Strip Parameters", "\\textbf{Strip Parameters}")

    return "\subsection{" + subtitle + """}
\includegraphics[width=0.40\\textwidth, valign=m]{referenceframes/""" + videoClassification + "/" + filename + """}
\includegraphics[width=0.60\\textwidth, valign=m]{eyepositiontraces/""" + videoClassification + "/" + plotname + """}\\\\
""" + "\n" + params + "\\newpage" + "\n\n"

def readImagePaths():
    referenceframespaths = {}
    for j in ["aoslo", "rodenstock_amblyopes", "rodenstock_amd", "rodenstock_normal", "tslo_amd", "tslo_normal"]:
        referenceframespaths[j] = []
        for i in os.listdir("referenceframes/" + j):
            if ".mat" not in i:
                referenceframespaths[j] += [i]

    return referenceframespaths

referenceframespaths = readImagePaths()

latex = ""
latex += header

for i in referenceframespaths:
    latex += section(i)
    for j in referenceframespaths[i]:
        latex += subsection(i, j)

latex += footer

file = open("analysis_report.tex", "w")
file.write(latex)
file.close()
