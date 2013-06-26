#!/usr/bin/ruby
# coding: utf-8

require './gradient_descent'
require 'matrix'

class Regression
  def initialize
    raise "Abstract class Regression cannot be instantiated"
  end

  def self.add_bias(x)
    if x.kind_of? Vector
      Vector[*x.to_a.unshift(1)]
    else
      Vector[1, x]
    end
  end

  def self.remove_bias(x)
    arr = x.to_a
    arr.shift
    Vector[*arr]
  end

  def self.hypothesis(parameters, x)
    raise "Abstract method self.hypothesis unimplemented"
  end

  def self.predict(parameters, x)
    hypothesis(parameters, add_bias(x))
  end

  def self.cost(parameters, xs, ys, norm_weight=0)
    raise "Abstract method self.cost unimplemented"
  end

  def self.cost_gradient(parameters, xs, ys, norm_weight=0)
    raise "Abstract method self.cost_gradient unimplemented"
  end

  def self.init_params(xs, ys)
    raise "Abstract method self.init_params unimplemented"
  end

  def self.gradient_descent(rate, xs, ys, monitor=nil, halt=nil)
    parameters = self.init_params(xs, ys)
    gd = GradientDescent.new(parameters, rate, &method(:cost_gradient))
    gd.each_iter(&monitor).stop_when(&halt).run
    gd.x
  end
end
