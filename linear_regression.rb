require './gradient_descent'
require 'matrix'

class LinearRegression
  attr_accessor :x_train, :y_train, :norm_weight, :parameters
  def initialize(x_train, y_train, norm_weight, parameters)
    @x_train = Array.new
    x_train.each_with_index do |x_i, i|
      # Add x[0] = 1 for all x_i
      @x_train.push(Vector[*x_train[i].to_a.unshift(1)])
    end
    @y_train = y_train
    @norm_weight = norm_weight
    @parameters = parameters
  end

  def evaluate features # assuming x[0] = 1
    features.inner_product(@parameters)
  end

  def cost parameters
    sum1 = 0
    sum2 = 0
    (0...@x_train.length).each do |i|
      sum1 += ((evaluate @x_train[i]) - @y_train[i]) ** 2
      if i != 0 # don't penalize bias parameter
        sum2 += parameters[i] ** 2
      end
    end
    normalization = @norm_weight / (2 * parameters.size) * sum2
    1 / (2 * @x_train.length) * sum1 + normalization
  end

  def cost_gradient parameters
    arr = []
    (0...@parameters.size).each do |j|
      sum = 0
      if j != 0
        sum = @norm_weight / parameters.size * @parameters[j]
      end
      (0...@x_train.length).each do |i|
        sum += (evaluate(@x_train[i]) - @y_train[i]) * @x_train[i][j]
      end
      arr.push(sum / parameters.size)
    end
    Vector[*arr]
  end

  def fit!(rate, halt, &monitor)
    gd = GradientDescent.new(method(:cost_gradient), @parameters, rate, halt)
    gd.run! do
      @parameters = gd.x
      monitor.call(gd)
    end
  end
end


if __FILE__ == $0
  # Simple test
  training_x = [Vector[1], Vector[2], Vector[3]]
  training_y = [0, 1, 2.1]
  lr = LinearRegression.new(training_x, training_y, 0.01, Vector[1, 2])
  lr.fit!(0.03, lambda {|gd| (gd.x - gd.step(gd.x)).magnitude < 0.00000001}) do |gd|
    if gd.iterations % 500 == 0
      puts "  Iteration %d: %s" % [gd.iterations, gd.x.to_s]
    end
  end
  puts "#{lr.parameters}"
end
