import pdb
import numpy
import theano
import theano.tensor as T

#print theano.config.floatX
N = 4
feats = 10

D = (numpy.random.randn(N, feats), numpy.random.randint(size=N,low=0,high=2))
training_steps = 100

x = T.matrix("x")
y = T.vector("y")

w = theano.shared(numpy.random.randn(feats), name="w")
b = theano.shared(numpy.zeros(()), name="b")

print "Initial model:"
print w.get_value()
print b.get_value()

#pdb.set_trace()

p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))
xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1)
cost = xent.mean() + 0.01 * (w ** 2).sum()
gw, gb = T.grad(cost, [w, b])
prediction = p_1 > 0.5

predict = theano.function(inputs=[x], outputs=prediction)
train = theano.function(inputs=[x,y], outputs=[prediction, xent], 
			updates=((w, w-0.1*gw), (b, b-0.1*gb)))

for i in range(training_steps):
	pred, err = train(D[0], D[1])

print "Final model:"
print w.get_value()
print b.get_value()
print "Target values for D: ", D[1]
print "Prediction on D: ", predict(D[0])
