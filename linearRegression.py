import numpy as np
import matplotlib.pyplot as plt
import time

st = time.time()
def recalculateMGradient():
    return (2*(yPredictions-y)*x).mean()

def recalculateBGradient():
    return (2*(yPredictions-y)).mean()

def plotData():
    plt.scatter(x, y, c='r')
    plt.plot(x, baseLine, linestyle='solid', c='r')
    plt.plot(x, m*x+b, linestyle='solid', c='b')
   #plt.show()



# Generate Random Data
numPoints =100
x = np.random.rand(numPoints)


randomB = np.random.randint(0, 100)
randomM = np.random.randint(0, 100) - 40
randomError = np.random.randint(1, 50)
baseLine = randomM*x+randomB

y = baseLine+(randomError*np.random.rand(numPoints))-(randomError/2)



# Generate Random Line
m = np.random.rand(1)
print(m)
b = np.random.rand(1)
yPredictions = m*x+b
initialLoss = ((yPredictions-y)**2).mean()


# mGradient = dloss/dm = d/dm[(yPredictions-y)**2] = 2*((m*x+b)-y)*x
# bGradient = dloss/db = d/db[(yPredictions-y)**2] = 2*((m*x+b)-y)*1
mGradient = recalculateMGradient()
bGradient = recalculateBGradient()



# Iterate and Adjust Prediction
alpha = 0.1
for i in range(1000):
    yPredictions = m*x+b
    mGradient = recalculateMGradient()
    bGradient = recalculateBGradient()
    m = m-alpha*mGradient
    b = b-alpha*bGradient



# Record Improved Loss and Output
finalLoss = (((m*x+b)-y)**2).mean()
print(
    f"Initial Loss: {initialLoss}\n"
    f"Final Loss: {finalLoss}\n"
)
plotData()
et = time.time()
print(et-st)