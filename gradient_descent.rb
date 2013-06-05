#!/usr/bin/ruby
# coding: utf-8


class GradientDescent
  attr_accessor :f_prime, :x, :rate, :halt
  attr_reader :iterations
  def initialize(f_prime, x, rate, halt)
    @f_prime = f_prime
    @x = x
    @rate = rate
    @halt = halt
    @iterations = 0
  end

  def step val
    val - @rate * @f_prime.call(val)
  end

  def update!
    @x = step @x
  end

  def run!
    while !@halt.call(self)
      if block_given?
        yield(self)
      update!
      @iterations += 1
      end
    end
  end
end

if __FILE__ == $0
  puts "* Testing gradient descent..."
  # second-degree polynomial
  puts "* Test 0 :: Minimizing f(x) = (x - 2)^2 - 3 | f'(x) = 2x - 4"
  f_prime = lambda {|x| 2 * x - 4}
  halt = lambda {|gd| gd.x - gd.step(gd.x) < 0.00000001}
gd = GradientDescent.new f_prime, 30, 0.002, halt
  monitor = lambda do |gd|
    if gd.iterations % 500 == 0
      puts "  Iteration #{gd.iterations}: #{gd.x.to_s}"
    end
  end
  gd.run! &monitor
  puts "! Minimum found at x = #{gd.x} after #{gd.iterations} iterations"

  # vector function
  require 'matrix'
  puts "* Test 1 :: Minimizing f(x0, x1, x2) = x0^2 + x1^2 + x2^2"
  puts "  | (del f)(x0, x1, x2) = <2x0, 2x1, 2x2>"
  f_prime = lambda {|x| Vector[2 * x[0], 2 * x[1], 2 * x[2]]}
  halt = lambda {|gd| (gd.x - gd.step(gd.x)).magnitude < 0.00000001}
  gd = GradientDescent.new f_prime, Vector[-35, 140, 2], 0.003, halt
  gd.run! &monitor
  puts "! Minimum found at x = #{gd.x.to_s} after #{gd.iterations} iterations"
end
