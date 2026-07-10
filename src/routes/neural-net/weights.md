# What's Inside weights.json

When you train a neural network, the only thing it "learns" is a set of
numbers called **weights** and **biases**. That's it. The entire intelligence
of the network is captured in these numbers. This file explains exactly what
they mean and how they're used to make predictions.

---

## Single-Layer Perceptron (2 -> 1)

A single-layer perceptron has two inputs and one output. It's the simplest
possible neural network — and it can't solve XOR.

### The weights file

```json
{
  "type": "single-layer",
  "w1": 0.847,
  "w2": -0.293,
  "bias": 0.156
}
```

Three numbers. That's the entire network.

### The architecture

```
  input 1 ----( w1 )----\
                          (+)---[ sigmoid ]---> output
  input 2 ----( w2 )----/
                  |
                bias
```

### How prediction works

To predict the output for an input like [1, 0]:

```
  Step 1: Multiply each input by its weight
          1 * 0.847 = 0.847
          0 * -0.293 = 0.000

  Step 2: Add them together with the bias
          0.847 + 0.000 + 0.156 = 1.003

  Step 3: Squeeze through sigmoid
          sigmoid(1.003) = 1 / (1 + e^(-1.003)) = 0.73
```

That's your prediction: **0.73**.

### What is sigmoid?

Sigmoid is an **activation function**. Without it, a neural network is
just multiplication and addition — which can only compute linear
relationships. Sigmoid introduces a curve, which is what lets networks
learn nonlinear patterns like XOR.

The formula:

```
  sigmoid(x) = 1 / (1 + e^(-x))
```

It takes any number from -infinity to +infinity and squashes it into
the range (0, 1). Here's what it looks like:

```
  output
  1.0 |                          ___________
      |                        /
      |                      /
  0.5 |                    /
      |                  /
      |                /
  0.0 |_______________/
      +----+----+----+----+----+----+----+----> input
         -6   -4   -2    0    2    4    6
```

Key properties:

- **Large positive input** (like 6) --> output near 1.0 (neuron "fires")
- **Large negative input** (like -6) --> output near 0.0 (neuron "silent")
- **Input near 0** --> output near 0.5 (neuron "unsure")

This is why it matters: after multiplying inputs by weights and adding
the bias, you might get any number. Could be -14, could be 203. Sigmoid
normalizes that into a clean 0-to-1 range you can interpret as a
probability or a yes/no signal.

Every neuron in this network does the same thing:

```
  1. Take inputs
  2. Multiply by weights
  3. Add bias
  4. Squeeze through sigmoid
  5. Pass result to the next layer
```

There are other activation functions (ReLU, tanh, GELU) used in modern
networks, but sigmoid is the classic — the one Rumelhart, Hinton, and
Williams used in 1986 when they proved backpropagation works.

### Why it fails at XOR

A single-layer perceptron draws a straight line through the input space.
XOR needs a curved boundary. No matter how you adjust w1, w2, and bias,
you can't draw a straight line that separates these four points correctly:

```
      x2
      1 |  target=1      target=0
        |     *              *
        |
      0 |  target=0      target=1
        |     *              *
        +------------------------  x1
             0              1
```

You need [0,1] and [1,0] on one side, [0,0] and [1,1] on the other.
Try it — there's no straight line that does it.

---

## Multi-Layer Network (2 -> 4 -> 1)

Add a **hidden layer** of 4 neurons between input and output, and
suddenly the network can learn XOR. This is the breakthrough that
backpropagation made possible.

### The weights file

```json
{
  "type": "multi-layer",
  "w1": [
    [5.82, -3.91, 4.17, -6.03],
    [5.79, -3.88, 4.21, -5.98]
  ],
  "b1": [-2.48, 5.93, -6.41, 2.71],
  "w2": [8.41, 7.53, -8.12, -8.67],
  "b2": -3.72
}
```

That's 21 numbers total. Let's break them down.

### The architecture

```
               HIDDEN LAYER
            .-- [ h0 ] --.
           /               \
  x1 ----+--- [ h1 ] ---+----(+)---[ sigmoid ]---> output
           \               /
  x2 ----+--- [ h2 ] ---+
           \               /
            '-- [ h3 ] --'

       w1 + b1           w2 + b2
     (2x4 = 8 weights   (4 weights
      + 4 biases)        + 1 bias)
```

### What each weight matrix means

**w1** (2x4 matrix) — connections from inputs to hidden neurons:

```
  w1[0] = [ 5.82, -3.91,  4.17, -6.03]   <-- weights FROM input x1
  w1[1] = [ 5.79, -3.88,  4.21, -5.98]   <-- weights FROM input x2
            |       |       |       |
            v       v       v       v
           h0      h1      h2      h3      <-- TO each hidden neuron
```

**b1** (4 values) — one bias per hidden neuron:

```
  b1 = [-2.48, 5.93, -6.41, 2.71]
          |      |      |      |
          v      v      v      v
         h0     h1     h2     h3
```

**w2** (4 values) — connections from hidden neurons to output:

```
  w2 = [ 8.41, 7.53, -8.12, -8.67]
          |      |      |      |
         h0     h1     h2     h3 ---> output
```

**b2** (1 value) — the output neuron's bias:

```
  b2 = -3.72 ---> added to output sum before sigmoid
```

### How prediction works

Let's predict the output for input [1, 1] (expected: 0).

```
  STEP 1: Compute each hidden neuron

    h0 = sigmoid( x1*w1[0][0] + x2*w1[1][0] + b1[0] )
       = sigmoid( 1*5.82 + 1*5.79 + (-2.48) )
       = sigmoid( 9.13 )
       = 0.9999

    h1 = sigmoid( 1*(-3.91) + 1*(-3.88) + 5.93 )
       = sigmoid( -1.86 )
       = 0.135

    h2 = sigmoid( 1*4.17 + 1*4.21 + (-6.41) )
       = sigmoid( 1.97 )
       = 0.878

    h3 = sigmoid( 1*(-6.03) + 1*(-5.98) + 2.71 )
       = sigmoid( -9.30 )
       = 0.0001

  STEP 2: Compute output

    output = sigmoid( h0*w2[0] + h1*w2[1] + h2*w2[2] + h3*w2[3] + b2 )
           = sigmoid( 0.9999*8.41 + 0.135*7.53 + 0.878*(-8.12) + 0.0001*(-8.67) + (-3.72) )
           = sigmoid( 8.41 + 1.02 + (-7.13) + 0.00 + (-3.72) )
           = sigmoid( -1.42 )
           = 0.19

  Prediction: 0.19 (close to 0) -- CORRECT
```

### What each hidden neuron learned

After training, each hidden neuron becomes a detector for a specific pattern:

```
  h0: fires when BOTH inputs are high  (AND-like)
  h1: fires when BOTH inputs are low   (NOR-like)
  h2: fires when at least one is high  (OR-like)
  h3: fires when BOTH inputs are high  (AND-like, inverted in output)
```

The output layer combines these detectors:

- h0 fires + h2 fires + h3 doesn't = both inputs on = output LOW (XOR = 0)
- h2 fires + h0 doesn't + h3 doesn't = one input on = output HIGH (XOR = 1)

The network decomposed an unsolvable problem into solvable sub-problems.
That's what hidden layers do.

---

## The key insight

The weights file IS the neural network. There's no code, no logic,
no rules — just numbers. The "intelligence" is entirely encoded in
how these numbers interact during the forward pass:

```
  multiply --> add --> sigmoid --> multiply --> add --> sigmoid --> answer
```

Every AI model works this way. GPT-4 has roughly a trillion of these
numbers instead of 21. But the principle is identical: data flows
forward through layers of weights, each layer transforming the signal,
until an answer comes out the other end.

Training is just finding the right numbers. The weights file is the result.
