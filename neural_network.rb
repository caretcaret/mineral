require './gradient_descent'
require 'matrix'

class NeuralNetwork
  attr_accessor :input_dim, :output_dim, :xs, :ys, :norm_weight, :weights, :activations
  # Constructs a fully connected feed-forward neural network.
  # Params:
  # +hidden+:: An array of integers, +hidden[i]+ representing
  # the number of non-bias nodes in hidden layer +i+.
  # +xs+:: +Array+ of +Vector+s as training examples,
  # without the bias feature +x[0] = 1+.
  # +ys+:: +Array+ of +Vector+s as training outputs.
  # +norm_weight+:: Normalization weight lambda.
  # +weights=:: The weights of each node to be assigned.
  # If +nil+, weights will be assigned randomly between -1 and 1.
  def initialize(hidden, xs, ys, norm_weight=0, weights=nil)
    raise "No examples given" unless xs.size > 1 or ys.size > 1
    raise "Length mismatch for xs, ys" unless xs.size == ys.size
    @input_dim = xs[0].size
    @output_dim = ys[0].size

    xs.each do |x|
      raise "Dimension mismatch for xs" unless x.size == @input_dim
    end
    @xs = xs

    ys.each do |y|
      raise "Dimension mismatch for ys" unless y.size == @output_dim
    end
    @ys = ys

    @norm_weight = norm_weight

    layers = [@input_dim] + hidden + [@output_dim]
    @weights = weights
    if weights.nil?
      @weights = (1...layers.size).map do |i|
        Matrix.build(layers[i], layers[i-1] + 1) { 2 * rand - 1 }
      end
    end

    @activations = []
    @deltas = []
  end

  def logistic(x)
    case x
    when Numeric
      1 / (1 + Math.exp(-x))
    when Vector
      x.map { |x_i| logistic(x_i) }
    end
  end

  def logistic_prime(x)
    case x
    when Numeric
      logistic(x) * (1 - logistic(x))
    when Vector
      x.map { |x_i| logistic_prime(x_i) }
    end
  end

  # Performs forward-propagation on the neural network.
  # Params:
  # +weights+:: Weights for the network.
  # +input+:: The input to feed into the input layer, assuming
  # the bias feature +x[0] = 1+ is NOT present.
  def forward(weights, input)
    @activations[0] = Vector[*input.to_a.unshift(1)]
    (0...weights.size).each do |layer|
      raw_vector = weights[layer] * activations[layer]
      activation = logistic(raw_vector)
      @activations[layer + 1] = Vector[*activation.to_a.unshift(1)]
    end
  end

  # Evaluates the neural network on the input.
  # Params:
  # +input+:: The input to feed into the input layer, assuming
  # the bias feature +x[0] = 1+ is NOT present.
  def evaluate(input)
    forward(@weights, input)
    last_layer = @activations[@weights.size].to_a
    last_layer.shift # remove leading 1 bias element
    Vector[*last_layer]
  end

  # Trains the neural network using back-propagation.
  # Params:
  # +rate+:: The learning rate alpha used in gradient descent.
  # +monitor+:: A Proc taking in the +GradientDescent+ engine
  # and is run on each iteration of gradient descent.
  # +halt+:: A lambda taking in the +GradientDescent+ engine
  # and returning true if the training should halt.
  def train(rate, monitor=nil, halt=nil)

  end
end

if __FILE__ == $0
  xs = ((-5...5).map do |i|
    (-5...5).map do |j|
      (-5...5).map do |k|
        Vector[i, j, k]
      end
    end
  end).flatten
  ys = xs.map do |v|
    Vector[v.magnitude > 3 ? 1 : 0,
           v[0] * v[1] > 1 ? 1 : 0]
  end
  ann = NeuralNetwork.new([3, 3, 3], xs, ys)
  ann.train(0.03)
end
