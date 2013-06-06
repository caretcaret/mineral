#!/usr/bin/ruby
# coding: utf-8

require './gradient_descent'
require 'matrix'

class LogisticRegression
  attr_accessor :xs, :ys, :norm_weight, :parameters

  def initialize(xs, ys, norm_weight, parameters=nil)
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

  def hypothesis(parameters, x)
    1 / (1 + Math.exp(-x.inner_product(parameters)))
  end

  def predict(features)
    hypothesis(@parameters, features)
  end

  def cost(parameters)
    error = 0
    (0...@xs.size).each do |i|
      h = hypothesis(parameters, @xs[i])
      error += @ys[i] * Math.log(h)
      error += (1 - @ys[i]) * Math.log(1 - h)
    end
    norm_penalty = 0
    (1...@dimension).each do |j|
      norm_penalty += parameters[j]**2
    end
    1 / @xs.size * (-error + norm_weight * norm_penalty / 2)
  end

  def cost_gradient(parameters)
    arr = []
    (0...@dimension).each do |j|
      sum = 0
      if j != 0
        sum = @norm_weight * parameters[j]
      end
      (0...@xs.length).each do |i|
        sum += (hypothesis(parameters, @xs[i]) - @ys[i]) * @xs[i][j]
      end
      arr.push(sum / @xs.size)
    end
    Vector[*arr]
  end

  def gradient_descent(rate, monitor=nil, halt=nil)
    gd = GradientDescent.new(@parameters, rate, &method(:cost_gradient))
    gd.each_iter(&monitor).stop_when(&halt).run
    @parameters = gd.x
    self
  end

end

if __FILE__ == $0
  puts "* Testing logistic regression..."
  puts "1 Testing simple data set [(-10, 0), (-5, 0), (-1, 0), (1, 0), (5, 1), (10, 1)]"
  puts "1 with rate=0.05, normalization weight=0.5"
  monitor = lambda { |gd|
    if gd.iterations % 500 == 0
      puts "  Iteration #{gd.iterations}, parameters = #{gd.x}"
    end
  }
  training_x = [Vector[-10], Vector[-5], Vector[-1], Vector[1], Vector[5], Vector[10]]
  training_y = [0, 0, 0, 1, 1, 1]
  lr = LogisticRegression.new(training_x, training_y, 0.5)
  lr.gradient_descent(0.05, monitor)
  puts "! parameters = #{lr.parameters}"
end

