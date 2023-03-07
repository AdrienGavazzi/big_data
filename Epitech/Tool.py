import os

class tool():
    def convert2html(name: str, outputName: str = "", isOnlyOutput: bool = False):
        print("Convert " + name + " to html")
        output = " --no-input" if isOnlyOutput else ""
        outputName = name.split(".")[0] + ".html" if outputName == "" else outputName + ".html"
        resp = os.system('jupyter nbconvert --to html ' + output + ' --output ' + outputName + ' ' + name)
        print(resp)