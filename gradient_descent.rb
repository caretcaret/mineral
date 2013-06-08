#!/usr/bin/ruby
# coding: utf-8

class GradientDescent
  attr_accessor :gradient, :x, :rate
  attr_reader :iterations, :prev_x
  @@default_on_iteration = proc { |gd| }
  @@default_halt = proc { |gd|
    (gd.x - gd.prev_x).magnitude < 0.0000001 && gd.iterations > 1
  }

  # Creates a gradient descent engine.
  # Params:
  # +x_init+:: The initial input value.
  # +rate+:: The rate alpha to descend the gradient.
  # +&gradient+:: A block representing the gradient function.
  def initialize(x_init, rate, &gradient)
    @gradient = gradient
    @x = x_init
    @prev_x = x_init
    @rate = rate
    @on_iteration = @@default_on_iteration
    @halt = @@default_halt
    @iterations = 0
    self
  end

  # Returns the value at the next step of gradient descent.
  # Params:
  # +val+:: The input value.
  def step(val)
    if val.kind_of? Array
      grad = @gradient.call(val)
      val.each_with_index.map { |x, i| x - @rate * grad[i] }
    else
      val - @rate * @gradient.call(val)
    end
  end

  # Updates the value +self.x+ after a step of gradient descent.
  def step!
    @prev_x = @x
    @x = step(@x)
    self
  end

  # Sets a block taking in a +GradientDescent+ engine
  # to run on each iteration of gradient descent.
  # Params:
  # +&on_iteration+:: The block. If set to nil, runs nothing.
  def each_iter(&on_iteration)
    if !on_iteration.nil? || block_given?
      @on_iteration = on_iteration
    else
      @on_iteration = @@default_on_iteration
    end
    self
  end

  # Sets a block taking in a +GradientDescent+ engine that returns
  # true if it should halt, and false otherwise.
  # Params:
  # +&halt+:: The block. If set to nil, halts when the magnitude of
  # the change in each iteration is less than 1e-7.
  def stop_when(&halt)
    if !halt.nil? || block_given?
      @halt = halt
    else
      @halt = @@default_halt
    end
    self
  end

  # Runs gradient descent and sets +self.x+ to the minimizing value.
  def run
    while !@halt.call(self)
      @on_iteration.call(self)
      step!
      @iterations += 1
    end
    self
  end
end

if __FILE__ == $0
  require 'matrix'
  monitor = lambda { |gd|
    if gd.iterations % 500 == 0
      puts "  Iteration #{gd.iterations}, x = #{gd.x}"
    end
  }
  puts "* Testing gradient descent..."

  # second-degree polynomial
  puts "1 Minimizing f(x) = (x - 2)^2 - 3 :: f'(x) = 2x - 4"
  puts "1 with rate=0.002"
  gd = GradientDescent.new(x_init=30, rate=0.002) do |x|
    2 * x - 4
  end
  gd.each_iter(&monitor).run
  puts "! Iteration #{gd.iterations}, x = #{gd.x}"

  # vector function
  puts "2 Minimizing f(x0, x1, x2) = x0^2 + x1^2 + x2^2"
  puts "2 :: (del f)(x0, x1, x2) = Vector[2x0, 2x1, 2x2]"
  puts "2 with rate=0.002"
  gd = GradientDescent.new(x_init=Vector[-35, 140, 2], rate=0.002) do |x|
    Vector[2 * x[0], 2 * x[1], 2 * x[2]]
  end
  gd.each_iter(&monitor).run
  puts "! Iteration #{gd.iterations}, x = #{gd.x}"
end
