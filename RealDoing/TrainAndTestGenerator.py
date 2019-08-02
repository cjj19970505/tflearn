import io
import numpy as np
trainPhFile = io.open(file=r'.\RealDoing\train_ph.txt', mode='x', encoding='utf8')
trainBinFile = io.open(file=r'.\RealDoing\train_bin.txt', mode='x', encoding='utf8')
testPhFile = io.open(file=r'.\RealDoing\test_ph.txt', mode='x', encoding='utf8')
testBinFile = io.open(file=r'.\RealDoing\test_bin.txt', mode='x', encoding='utf8')
dispPhFile = io.open(file=r'.\RealDoing\disp_ph.txt', mode='x', encoding='utf8')
dispBinFile = io.open(file=r'.\RealDoing\disp_bin.txt', mode='x', encoding='utf8')

allPhFile = io.open(r'.\RealDoing\all_ph.txt', encoding='utf8')
allBinFile = io.open(r'.\RealDoing\all_bin.txt', encoding='utf8')
lineNum = 0
while True:
    phLine = allPhFile.readline()
    binLine = allBinFile.readline()
    if not(phLine and binLine):
        break
    toTrainFile = bool(np.random.binomial(1, 0.8, 1)[0])
    if toTrainFile:
        trainPhFile.writelines(phLine)
        trainBinFile.writelines(binLine)
    else:
        testPhFile.writelines(phLine)
        testBinFile.writelines(binLine)
        dispPhFile.writelines(phLine)
        dispBinFile.writelines(binLine)
print("Generate train and test complete")
