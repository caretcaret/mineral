#!/usr/bin/ruby
# coding: utf-8

require './gradient_descent'
require 'matrix'

class SoftmaxRegression
  attr_accessor :xs, :ys, :norm_weight, :parameters

  # Creates a new +SoftmaxRegression+ engine.
  # Params:
  # +xs+:: The training features (without the +1 bias feature), an array of vectors.
  # +ys+:: The training outputs, an array with elements +0..k-1+, where +k+ is the number of classes.
  # +norm_weight+:: Normalization weight lambda. Defaults to 0.
  # +parameters+:: Initialize the coefficient matrix with dimensions +k * (n+1)*, where +k+ is
  # the number of classes and +n+ is the number of training features. If set to nil, the values
  # will be initialized to a random matrix with elements between -1 and 1.
  def initialize(xs, ys, norm_weight=0, parameters=nil)
    raise "No examples given" unless xs.size > 0 || ys.size > 0
    raise "Length mismatch for xs, ys" unless xs.size == ys.size
    @dimension = xs[0].size + 1

    @xs = []
    xs.each_with_index do |x_i|
      # Add x_i[0] = 1 for all x_i
      x_new = Vector[*x_i.to_a.unshift(1)]
      raise "Dimension mismatch for xs" unless x_new.size == @dimension
      @xs.push(x_new)
    end
    @ys = ys
    # assuming ys is a list of classes 0..n-1
    @num_classes = ys.max + 1
    @norm_weight = norm_weight

    if parameters.nil?
      @parameters = Matrix.build(@num_classes, @dimension) { 2 * rand - 1 }
    else
      @parameters = parameters
    end

    raise %Q{Parameters dimension mismatch: parameters are
      #{@parameters.row_size} * #{@parameters.column_size},
      it should be #{@num_classes} * #{@dimension}} unless
      [@parameters.row_size, @parameters.column_size] == [@num_classes, @dimension]
  end

  # Returns the hypothesis h_theta(x).
  # Params:
  # +parameters+:: The coefficient matrix theta.
  # +x+:: The feature vector (including +1 bias) to evaluate the hypothesis on.
  def hypothesis(parameters, x)
    log_hypothesis(parameters, x).map do |log_component|
      Math.exp(log_component)
    end
  end

  # Returns the logarithm of the hypothesis h_theta(x). Used to prevent
  # overflow errors. The parameters are the same as in +hypothesis(parameters, x)+.
  def log_hypothesis(parameters, x)
    log_components = parameters.row_vectors.map do |parameter|
      parameter.inner_product(x)
    end
    # Normalize the largest value to 0, then add this factor back in
    # after taking the exponential and the log.
    m = log_components.max
    sum = 0
    log_components.each do |log_component|
      sum += Math.exp(log_component - m)
    end
    log_sum = m + Math.log(sum)
    # components / sum becomes log_components - sum
    log_probabilities = log_components.map do |log_component|
      log_component - log_sum
    end
    Vector[*log_probabilities]
  end

  # Returns the probability distribution given the features, where
  # the parameters are taken from the +SoftmaxRegression+ engine.
  # Params:
  # +features+:: Feature vector, where the bias feature x[0] = 1 has NOT been added.
  def predict(features)
    hypothesis(@parameters, Vector[*features.to_a.unshift(1)])
  end

  # Returns the normalized cost J(theta) evaluated at theta.
  # Params:
  # +parameters+:: The coefficient matrix theta.
  # +norm_weight+:: The normalization weight lambda. If not specified, defaults
  # to the normalization weight set in the +SoftmaxRegression+ engine.
  def cost(parameters, norm_weight=@norm_weight)
    error = 0
    normalization = 0
    @xs.each_with_index do |x, i|
      error += log_hypothesis(parameters, x)[ys[i]]
    end
    parameters.each_with_index do |elem, row, col|
      if col != 0
        normalization += elem ** 2
      end
    end
    -1.0 / @xs.size * error + norm_weight / 2.0 * normalization
  end

  # Returns the gradient of the cost function evaluated at theta, given as a +k * (n+1)+ matrix.
  # Params:
  # +parameters+:: The coefficient matrix theta.
  # +norm_weight+:: The normalization weight lambda. If not specified, defaults
  # to the normalization weight set in the +SoftmaxRegression+ engine.
  def cost_gradient(parameters, norm_weight=@norm_weight)
    # initialize to all 0s
    gradient = Matrix.build(@num_classes, @dimension) { 0 }
    
    @xs.each_with_index do |x, i|
      result = hypothesis(parameters, x)
      gradient += result * x.covector
      gradient -= Matrix.build(@num_classes, @dimension) do |klass, feature|
        if @ys[i] == klass
          x[feature]
        else
          0
        end
      end
    end
    normalization = Matrix.build(@num_classes, @dimension) do |klass, feature|
      if feature == 0
        0
      else
        norm_weight * parameters[klass, feature]
      end
    end
    gradient / @xs.size + normalization
  end

  # Runs gradient descent to set +self.parameters+ to the value that minimizes
  # the cost function.
  # Params:
  # +rate+:: The learning rate alpha for gradient descent.
  # +monitor+:: A lambda taking in a +SoftmaxRegression+ engine and is run on each
  # loop of gradient descent. If +nil+, does nothing.
  # +halt+:: A lambda taking in a +SoftmaxRegression+ engine and returns +true+ if it
  # should halt, and +false+ otherwise.
  def gradient_descent(rate, monitor=nil, halt=nil)
    gd = GradientDescent.new(@parameters, rate, &method(:cost_gradient))
    gd.each_iter(&monitor).stop_when(&halt).run
    @parameters = gd.x
    self
  end
end

if __FILE__ == $0
  puts "* Training softmax regression..."
  puts "1 Testing simple classes: x < -25, -25 <= x <= 25, 25 < x"
  puts "1 with rate=1, normalization weight=0.0"
  monitor = lambda { |gd|
    if gd.iterations % 500 == 0
      puts "  Iteration #{gd.iterations}, parameters = #{gd.x}"
    end
  }
  halt = lambda { |gd| gd.iterations > 10000 }
  training_x = (0...100).map {|i| Vector[i - 50] }
  training_y = training_x.map do |x|
    if x[0] < -25
      0
    elsif x[0] > 25
      2
    else
      1
    end
  end
  sr = SoftmaxRegression.new(training_x, training_y, 0.0)
  sr.gradient_descent(1, monitor, halt)
  puts "! parameters = #{sr.parameters}"
  puts "! Prediction for 25.5: #{sr.predict(Vector[25.5])}"
end
