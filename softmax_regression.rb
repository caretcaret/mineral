#!/usr/bin/ruby
# coding: utf-8

require './gradient_descent'
require 'matrix'

class SoftmaxRegression
  attr_accessor :xs, :ys, :norm_weight, :parameters

  def initialize(xs, ys, norm_weight=0, parameters=nil)
    raise "No examples given" unless xs.size > 1 || ys.size > 1
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
    @num_classes = ys.max
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

  def hypothesis(parameters, x)
    sum = 0
    hypothesis_arr = []
    parameters.row_vectors.each do |parameter|
      component = Math.exp(parameter.inner_product(x))
      sum += component
      hypothesis_arr.push(component)
    end
    1.0 / sum * Vector[*hypothesis_arr]
  end

  def predict(features)
    hypothesis(@parameters, features)
  end

  def cost(parameters)
    sum = 0
    @xs.each_with_index do |x, i|
      sum += Math.log(hypothesis(parameters, x)[ys[i]])
    end
    -1.0 / @xs.size * sum
  end

  def cost_gradient(parameters)
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
    gradient / @xs.size
  end

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
  puts "1 with rate=0.05, normalization weight=0.5"
  monitor = lambda { |gd|
    if gd.iterations % 500 == 0
      puts "  Iteration #{gd.iterations}, parameters = #{gd.x}"
    end
  }
  halt = lambda { |gd|
    gd.iterations > 10000
  }
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
  sr = SoftmaxRegression.new(training_x, training_y, 0.5)
  sr.gradient_descent(0.0005, monitor, halt)
  puts "! parameters = #{sr.parameters}"
end