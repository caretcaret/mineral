#!/usr/bin/ruby
# coding: utf-8

require './gradient_descent'
require 'matrix'

class LinearRegression
  attr_accessor :xs, :ys, :norm_weight, :parameters

  # Creates a new +LinearRegression+ engine.
  # Params:
  # +xs+:: Training features (without the +1 bias feature), an array of vectors.
  # +ys+:: Training outputs, an array of Fixnums.
  # +norm_weight+:: Normalization weight lambda.
  # +parameters+:: Initial coefficients vector theta with dimension equal to
  # the training features vectors (with the +1 bias feature). If set to nil, the values
  # will be initialized to a random vector with elements between -10 and 10.
  def initialize(xs, ys, norm_weight=0, parameters=nil)
    raise "No examples given" unless xs.size > 1 or ys.size > 1
    raise "Length mismatch for xs, ys" unless xs.size == ys.size
    @dimension = xs[0].size + 1

    @xs = Array.new
    xs.each_with_index do |x_i|
      # Add x[0] = 1 for all x_i
      x_new = Vector[*x_i.to_a.unshift(1)]
      raise "Dimension mismatch for xs" unless x_new.size == @dimension
      @xs.push(x_new)
    end
    @ys = ys
    @norm_weight = norm_weight

    if parameters.nil?
      @parameters = Vector[*@dimension.times.map {-20 * Random.rand + 10}]
    else
      @parameters = parameters
    end
    raise "Parameter dimension mismatch: parameters.size = #{@parameters.size},
    dimension = #{@dimension}" unless @parameters.size == @dimension
  end

  # Returns the hypothesis h_theta(x) = theta^T x.
  # Params:
  # +parameters+:: The coefficient vector theta.
  # +x+:: The feature vector to evaluate the hypothesis on, assuming x[0] = 1.
  def hypothesis(parameters, x) # assuming x[0] = 1
    x.inner_product(parameters)
  end

  # Returns the prediction h_theta(x), where theta is taken
  # from the value set in the linear regression engine.
  # Params:
  # +features+:: Feature vector, where the bias feature x[0] = 1 has NOT been added.
  def predict(features)
    hypothesis(@parameters, Vector[*features.to_a.unshift(1)])
  end

  # Returns the square error and normalized cost J(theta) evaluated at theta.
  # Params:
  # +parameters+:: The coefficient vector theta.
  # +norm_weight+:: The normalization weight lambda. If not specified, takes the
  # normalization weight set in the linear regression engine.
  def cost(parameters, norm_weight=@norm_weight)
    error = 0
    (0...@xs.size).each do |i|
      error += (hypothesis(parameters, @xs[i]) - @ys[i])**2
    end
    norm_penalty = 0
    (1...@dimension).each do |j|
      norm_penalty += parameters[j]**2
    end
    1 / (2 * @xs.size) * (error + norm_weight * norm_penalty)
  end

  # Returns the gradient of the cost function, evaluated at theta.
  # Params:
  # +parameters+:: The coefficient vector theta.
  # +norm_weight+:: The normalization weight lambda. If not specified, takes the
  # normalization weight set in the linear regression engine.
  def cost_gradient(parameters, norm_weight=@norm_weight)
    arr = []
    (0...@dimension).each do |j|
      sum = 0
      if j != 0
        sum = norm_weight * parameters[j]
      end
      (0...@xs.length).each do |i|
        sum += (hypothesis(parameters, @xs[i]) - @ys[i]) * @xs[i][j]
      end
      arr.push(sum / @xs.size)
    end
    Vector[*arr]
  end

  # Runs gradient descent to set +self.parameters+ to the value that minimizes
  # the cost function.
  # Params:
  # +rate+:: The learning rate alpha for gradient descent.
  # +monitor+:: A lambda taking in a +GradientDescent+ engine
  # run on each loop of gradient descent. If nil, does nothing.
  # +halt+:: A lambda taking in a +GradientDescent+ engine and returning
  # true when it should halt, and false otherwise. If set to nil, uses
  # the default +GradientDescent+ halting behavior.
  def gradient_descent(rate, monitor=nil, halt=nil)
    gd = GradientDescent.new(@parameters, rate, &method(:cost_gradient))
    gd.each_iter(&monitor).stop_when(&halt).run
    @parameters = gd.x
    self
  end
end

if __FILE__ == $0
  puts "* Testing linear regression..."
  puts "1 Testing simple data set [(1, 0), (2, 1), (3, 2)]"
  puts "1 with rate=0.03, normalization weight=0"
  monitor = lambda { |gd|
    if gd.iterations % 500 == 0
      puts "  Iteration #{gd.iterations}, parameters = #{gd.x}"
    end
  }
  training_x = [Vector[1], Vector[2], Vector[3]]
  training_y = [0, 1, 2]
  lr = LinearRegression.new(training_x, training_y, 0)
  lr.gradient_descent(0.03, monitor)
  puts "! parameters = #{lr.parameters}"
end
