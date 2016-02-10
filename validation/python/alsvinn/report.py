import numpy
import pylab
class Report(object):
    def __init__(self, outtitle, title, intro):
        self.text = "\\section{%s}\n\n%s\n\n\n" % (title, intro)
        self.outtitle = outtitle
        self.image_counter = 0
        self.title = title


        pylab.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
        pylab.rcParams['savefig.dpi'] = 300
        pylab.rc('text', usetex=True)

    def add_convergence_plot(self, resolutions, convergence_data):
        resolutions = [resolution[0] for resolution in resolutions]
        self.start_figure("Convergence plot for %s." % self.title)
        for (n, convergence) in enumerate(convergence_data["convergence"]):
            rate = 100000
            for (key, yvalues) in convergence.iteritems():
                pylab.loglog(resolutions, yvalues, label=key)
                rate = min(rate, -numpy.polyfit(numpy.log(resolutions), numpy.log(yvalues), 1)[0])
            pylab.loglog(resolutions, resolutions**(-rate), "--", label="$\Delta x^{%1.2f}$" % round(rate, 2))
            pylab.grid("on")
            pylab.legend()
            self.add_subplot("%s" % (self.map_to_str(convergence_data["configurations"][n])))
            pylab.close()
        self.end_figure()

    def map_to_str(self, map):
        string = ""
        for (key, value) in map.iteritems():
            string += "\\textbf{%s}:~%s, " % (key, value)
        string = string[:-2]
        return string
    def start_figure(self, caption):
        self.text += "\\begin{figure}[h]\n\caption{%s}\n" % caption
    def end_figure(self):
        self.text += "\n\\end{figure}\n"

    def add_plot_data(self, plot_data, text):
        for (key, xy) in plot_data.iteritems():
            x = xy[0]
            y = xy[1]
            pylab.plot(x, y, label=key)
        pylab.legend()
        self.add_plot(text)
        pylab.close()

    def save_plot(self):
        outname= "%s_%d.png" % (self.outtitle, self.image_counter)
        pylab.savefig(outname)

        self.image_counter += 1

        return outname

    def add_subplot(self, text):
        outname = self.save_plot()

        self.text +="\\begin{subfigure}[b]{0.5\\textwidth}\n\\includegraphics[width=\\textwidth]{%s}\n\\caption{%s}\n\\end{subfigure}\n" % (outname, text)

    def add_plot(self,text):
        outname = self.save_plot()

        self.text += "\\begin{figure}[h]\n\includegraphics[width=0.6\\textwidth]{%s}\\caption{%s}\\end{figure}\n" % (outname, text)

    def write(self, outfile):
        with open(outfile, "w") as f:
            f.write("\\documentclass{article}\n")

            f.write("\\usepackage{amsmath,amssymb,amsfonts,graphicx, caption, subcaption}\n")
            f.write("\\begin{document}")
            f.write(self.text)
            f.write("\\end{document}")
