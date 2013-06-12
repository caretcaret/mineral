#!/usr/bin/ruby
# coding: utf-8

require './gradient_descent'
require 'matrix'

class NeuralNetwork
  attr_accessor :xs, :ys, :norm_weight, :weights
  attr_reader :input_dim, :output_dim, :activations, :deltas
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
    # weights is indexed by source layer; weights[i] =>
    # the weights between layer i and i+1, where layers are 0-indexed
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
      1.0 / (1.0 + Math.exp(-x))
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

  def elementwise_product(u, v)
    u.map2(v) do |u_i, v_i|
      u_i * v_i
    end
  end

  def add_bias_elem(v)
    Vector[*v.to_a.unshift(1)]
  end

  def remove_bias_elem(v)
    arr = v.to_a
    first = arr.shift
    return Vector[*arr]
  end

  # Performs forward-propagation on the neural network. Returns
  # the activation matrices in an array indexed by +layer - 1+.
  # Params:
  # +weights+:: Weights for the network.
  # +input+:: The input to feed into the input layer, assuming
  # the bias feature +x[0] = 1+ is NOT present.
  def forward(weights, input)
    activations = []
    activations[0] = add_bias_elem(input)
    (0...weights.size).each do |layer|
      preactivation = weights[layer] * activations[layer]
      activation = logistic(preactivation)
      activations[layer + 1] = add_bias_elem(activation)
    end
    # the activations are indexed by layer (0-indexed)
    # and include the +1 bias unit
    activations
  end

  # Same as +forward+, but sets +self.activations+ to the activations.
  def forward!(input)
    @activations = forward(@weights, input)
  end

  # Evaluates the neural network on the input and returns the
  # output.
  # Params:
  # +weights+:: The weights to be used for the neural network.
  # +input+:: The input to feed into the input layer, assuming
  # the bias feature +x[0] = 1+ is NOT present.
  def evaluate(weights, input)
    activations = forward(weights, input)
    remove_bias_elem(activations[weights.size])
  end

  # Same as +evaluate+, but uses and sets all intermediate internal values.
  def evaluate!(input)
    forward!(input)
    remove_bias_elem(@activations[@weights.size])
  end

  # Performs backward-propagation on the neural network. Returns
  # the delta vectors in an array indexed by +layer - 1+.
  # Params:
  # +weights+:: Weights for the network.
  # +activations+:: Activations for the network.
  # +output+:: The "correct" outputs corresponding to the +activations+.
  def backward(weights, activations, output)
    delta = []
    # delta[i] represents the error in layer i caused by layer i - 1
    # => layer 0 does not have a delta because it is the input layer
    delta[weights.size] = remove_bias_elem(activations[weights.size]) - output
    (weights.size - 1).downto(1).each do |layer|
      # layer = the layer that we want to calculate deltas for
      reverse_error = weights[layer].transpose * delta[layer + 1]
      reverse_gradient = activations[layer].map { |a| a * (1 - a) }
      delta[layer] = elementwise_product(reverse_error, reverse_gradient)
      delta[layer] = remove_bias_elem(delta[layer])
    end
    delta.shift # remove nil in delta layer 0
    delta
  end

  # Same as +backward+, but sets +self.deltas+ to the deltas.
  def backward!(output)
    @deltas = backward(@weights, @activations, output)
  end

  # Returns the gradient of the neural network cost function
  # evaluated at the given weights.
  # Params:
  # +weights+:: Weights for the network.
  def cost_gradient(weights)
    big_delta = []
    (0...@xs.size).each do |i|
      activations = forward(weights, @xs[i])
      delta = backward(weights, activations, @ys[i])
      (0...weights.size).each do |layer|
        if big_delta[layer].nil?
          big_delta[layer] = delta[layer] * activations[layer].covector
        else
          big_delta[layer] += delta[layer] * activations[layer].covector
        end
      end
    end
    gradient = []
    (0...big_delta.size).each do |l|
      rows = big_delta[l].row_size
      cols = big_delta[l].column_size
      gradient[l] = Matrix.build(rows, cols) do |row, col|
        norm_factor = col == 0 ? 0 : @norm_weight
        big_delta[l][row, col] / @xs.size + norm_factor * weights[l][row, col]
      end
    end
    gradient
  end

  # Returns the cost function evaluated at the given weights. Used for
  # gradient checking.
  # Params:
  # +weights+:: The weights for the network.
  def cost(weights)
    error = 0
    normalization = 0
    (0...@xs.size).each do |i|
      result = evaluate(weights, @xs[i])
      (0...@output_dim).each do |k|
        error += @ys[i][k] * Math.log(result[k]) + (1 - @ys[i][k]) * Math.log(1 - result[k])
      end
    end
    weights.each do |layer_matrix|
      (0...layer_matrix.row_size).each do |i|
        (1...layer_matrix.column_size).each do |j|
          normalization += layer_matrix[i,j]**2
        end
      end
    end
    -error / @xs.size + (@norm_weight / 2.0) * normalization
  end

  # Slow cost gradient approximation using definition of derivative. Used for gradient checking.
  # Params:
  # +weights+:: The weights for the network.
  # +epsilon+:: The level of approximation. Defaults to 1e-8.
  def slow_cost_gradient(weights, epsilon=0.00000001)
    gradient = []
    (0...weights.size).each do |layer|
      # save the matrix
      weight_matrix = weights[layer]
      weight_arr = weights[layer].to_a
      # make mutable matrix by editing the corresponding array
      weights[layer] = Matrix.rows(weight_arr, copy=false)
      gradient[layer] = Matrix.build(weight_matrix.row_size, weight_matrix.column_size) do |row, col|
        weight_arr[row][col] += epsilon
        c1 = cost(weights)
        weight_arr[row][col] -= 2 * epsilon
        c2 = cost(weights)
        weight_arr[row][col] += epsilon
        (c1 - c2) / (2.0 * epsilon)
      end
      # restore the matrix
      weights[layer] = weight_matrix
    end
    gradient
  end

  # Trains the neural network using back-propagation.
  # Params:
  # +rate+:: The learning rate alpha used in gradient descent.
  # +monitor+:: A Proc taking in the +GradientDescent+ engine
  # and is run on each iteration of gradient descent.
  # +halt+:: A lambda taking in the +GradientDescent+ engine
  # and returning true if the training should halt.
  def train(rate, monitor=nil, halt=nil)
    gd = GradientDescent.new(@weights, rate, &method(:cost_gradient))
    if halt.nil?
      halt = lambda { |gd| gd.iterations > 1000 }
    end
    gd.each_iter(&monitor).stop_when(&halt).run
    @weights = gd.x
    self
  end
end

if __FILE__ == $0
  puts "* Testing neural network..."
  puts "1 Training examples: points (x, y, z) with -4.5 <= x, y, z <= 4.5"
  puts "1 and x^2, y^2, z^2 terms"
  puts "1 Training outputs: if the point is at least 4 from the origin,"
  puts "1 and if the point is at least 4 from the origin when projected on z = 0"
  puts "1 Architecture: no hidden layers"
  puts "1 Ideal weights: Matrix[[-16, 0, 0, 0, 1, 1, 1], [-16, 0, 0, 0, 1, 1, 0]]"
  xs = ((-5...5).map do |i|
    (-5...5).map do |j|
      (-5...5).map do |k|
        Vector[i + 0.5, j + 0.5, k + 0.5, (i+0.5)**2, (j+0.5)**2, (k+0.5)**2]
      end
    end
  end).flatten
  ys = xs.map do |v|
    Vector[Math.sqrt(v[0]**2 + v[1]**2 + v[2]**2) > 4 ? 1 : 0,
           Math.sqrt(v[0]**2 + v[1]**2) > 4 ? 1 : 0]
  end
  ann = NeuralNetwork.new([], xs, ys, 0.0)
  $stdout.sync = true
  print "  Training... iteration"
  monitor = lambda { |gd|
    if gd.iterations % 20 == 0
      print " #{gd.iterations}"
    end
  }

  ann.train(1, monitor)
  puts "\n! Weights: #{ann.weights}"
  puts "  Testing accuracy..."
  a = b = c = d = 0
  (0...xs.size).each do |i|
    result = ann.evaluate!(xs[i])
    if ys[i][0] == result[0].round && ys[i][1] == result[1].round
      a += 1
    elsif ys[i][0] == result[0].round && ys[i][1] != result[1].round
      b += 1
    elsif ys[i][0] != result[0].round && ys[i][1] == result[1].round
      c += 1
    else
      d += 1
    end
  end
  puts "! Both correct: #{a}"
  puts "! Second output incorrect: #{b}"
  puts "! First output incorrect: #{c}"
  puts "! Both incorrect: #{d}"

  # gradient checking
  # ann.norm_weight = 1
  # a = ann.cost_gradient(ann.weights)
  # b = ann.slow_cost_gradient(ann.weights)
  # p a
  # p b
  # (0...a.size).each do |i|
  #   (0...a[i].row_size).each do |j|
  #     (0...a[i].column_size).each do |k|
  #       puts (a[i][j, k] - b[i][j, k]) / b[i][j, k]
  #     end
  #   end
  # end

end
