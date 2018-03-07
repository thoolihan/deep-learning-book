import os

def open_plot(plot_file):
    if os.name == "posix":
        os_name = os.popen("uname -a").read()
        if "darwin" in os_name.lower():
            os.system("open {}".format(plot_file))