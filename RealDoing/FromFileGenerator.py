import io
import numpy as np
class FromFileGenerator:
    def __init__(self, phFilePath, binFilePath):
        self.PhFilePath = phFilePath
        self.BinFilePath = binFilePath
    
    def GetNext(self):
        phFile = io.open(self.PhFilePath, encoding='utf8')
        binFile = io.open(r'.\RealDoing\train_bin.txt', encoding='utf8')
        while True:
            x_file_line = phFile.readline()
            y_file_line = binFile.readline()
            if not(x_file_line and y_file_line):
                break
            xdatalist = x_file_line.split()
            ydatalist = y_file_line.split()
            yield (np.asarray(xdatalist, dtype=np.float32), np.asarray(ydatalist, dtype=np.int))