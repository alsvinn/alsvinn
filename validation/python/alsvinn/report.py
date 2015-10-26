import numpy
import pylab
class Report(object):
    def __init__(self, outtitle, title, intro):
        self.text = "\\section{%s}\n\n%s" % (title, intro)
        self.outtitle = outtitle
        self.image_counter = 0
        self.title = title

    def add_convergence_plot(self, resolutions, convergence_data):
        resolutions = [resolution[0] for resolution in resolutions]
        for (n, convergence) in enumerate(convergence_data["convergence"]):

            rate = 100000
            for (key, yvalues) in convergence.iteritems():
                pylab.loglog(resolutions, yvalues, label=key)
                pylab.legend()
                pylab.grid("on")
                print yvalues
                print resolutions
                rate = min(rate, -numpy.polyfit(numpy.log(resolutions), numpy.log(yvalues), 1)[0])
            pylab.loglog(resolutions, resolutions**(-rate), "--")
            self.add_plot("Rate $\\approx %f$. Convergence plot for %s with configuration %s" % (rate, self.title, str(convergence_data["configurations"][n])))
            pylab.close()

    def add_plot(self,text):
        outname= "%s_%d.png" % (self.outtitle, self.image_counter)
        pylab.savefig(outname)

        self.image_counter += 1

        self.text += "\\begin{figure}\includegraphics[width=0.6\\textwidth]{%s}\caption{%s}\\end{figure}" % (outname, text)

    def write(self, outfile):
        with open(outfile, "w") as f:
            f.write("\\documentclass{article}\n")

            f.write("\\usepackage{amsmath,amssymb,amsfonts,graphicx}\n")
            f.write("\\begin{document}")
            f.write(self.text)
            f.write("\\end{document}")
