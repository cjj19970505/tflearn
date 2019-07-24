trainFigure = plt.figure(1)
for i in range(16):
    image = mnist.train.images[i].reshape([28,28])
    subPlot = trainFigure.add_subplot(4,4,i+1)
    subPlot.imshow(image)
    #values.index(max(values))
    plt.title("TrainNo."+str(i+1) + " Label:"+str(np.argmax(mnist.train.labels[i])))

testFigure = plt.figure(2)
for i in range(16):
    image = mnist.test.images[i].reshape([28,28])
    subPlot = testFigure.add_subplot(4,4,i+1)
    subPlot.imshow(image)
    plt.title("TestNo."+str(i+1))

#plt.show(block = False)