#!/usr/bin/ruby
# coding: utf-8

require 'matrix'

class Preprocessor
  attr_accessor :xs, :ys, :history

  # Creates a new preprocessor. Access the processed values with +Preprocessor#xs+
  # and +Preprocessor#ys+.
  # Params:
  # +xs+:: The input values or vectors for the ML problem.
  # +ys+:: The output values or vectors for the ML problem. If not specified,
  # defaults to the empty list, which does nothing.
  def initialize(xs, ys=[])
    raise "No examples given" unless xs.size > 1
    raise "Length mismatch for xs, ys" unless ys.empty? || xs.size == ys.size
    @dimension = xs[0].size
    @xs = xs
    @xs.each do |x|
      raise "Dimension mismatch for xs" unless x.size == @dimension
    end
    @ys = ys
    @history = []
  end

  # Using the equality operator, assigns output classes +ys+ to integers between
  # 0 and k, where k is the number of distinct classes.
  def normalize_classes
    classes = [] # maps integers to items
    len = 0
    @ys.map! do |y|
      index = classes.find_index(y)
      if index.nil?
        classes << y
        index = len
        len += 1
      end
      index
    end

    @history << [:normalize_classes, classes]
    self
  end

  # Returns the mean-normalized value.
  # Params:
  # +x+:: The value (vector or numeric).
  # +mean+:: The mean (type must match that of +x+).
  def pack_mean(x, mean)
    x - mean
  end

  # Returns the original value given the mean-normalized value.
  # +x+:: The value (vector or numeric).
  # +mean+:: The mean (type must match that of +x+).
  def unpack_mean(x, mean)
    x + mean
  end

  # Mean-normalizes all +xs+.
  def normalize_mean_x
    mean_x = mean(@xs)
    @xs.map! { |x| pack_mean(x, mean_x) }
    @history << [:normalize_mean_x, mean_x]
    self
  end

  # Mean-normalizes all +ys+.
  def normalize_mean_y
    mean_y = mean(@ys)
    @ys.map! { |y| pack_mean(y, mean_y) }
    @history << [:normalize_mean_y, mean_y]
    self
  end

  # Scales a value.
  # Params:
  # +x+:: The value to scale.
  # +sd+:: The standard deviation (vector or numeric) to scale it by.
  def pack_scale(x, sd)
    if x.kind_of? Vector
      x.map2(sd) do |x_i, sd_i|
        if sd_i == 0
          # if standard deviation is 0, then the normalized features
          # must all be 0. Just set it to 0
          0
        else
          # otherwise divide by sd to scale the feature
          x_i / sd_i
        end
      end
    else
      sd == 0 ? 0 : x / sd
    end
  end

  # Unscales a value.
  # Params:
  # +x+:: The value to scale.
  # +sd+:: The standard deviation (vector or numeric) to scale it by.
  def unpack_scale(x, sd)
    if x.kind_of? Vector
      x.map2(sd) { |x_i, sd_i| x_i * sd_i }
    else
      x * sd
    end
  end

  # Scales all +xs+.
  def scale_x
    sd = sdev(@xs)
    @xs.map! { |x| pack_scale(x, sd) }
    @history << [:scale_x, sd]
    self
  end

  # Scales all +ys+.
  def scale_y
    sd = sdev(@ys)
    @ys.map! { |y| pack_scale(y, sd) }
    @history << [:scale_y, sd]
    self
  end

  # Performs mean-normalization and feature scaling on +xs+.
  # Params:
  # +regression+:: If true, will standardize +ys+ as well as +xs+.
  # If false, will standardize +xs+ and normalize classes for +ys+.
  def standardize(regression=true)
    normalize_mean_x
    scale_x

    if regression
      normalize_mean_y
      scale_y
    else
      normalize_classes
    end
    self
  end

  # Finds the mean of a list of vectors or numbers.
  # +list+:: The list.
  def mean(list)
    if list[0].kind_of? Vector
      sum = Vector[*Array.new(list[0].size) { 0 }]
    else
      sum = 0
    end
    list.each do |elem|
      sum += elem
    end
    1.0 / list.size * sum
  end

  # Finds the standard deviation of a list of vectors or numbers.
  # Params:
  # +normalized+:: A list of vectors or numbers whose mean is 0.
  def sdev(normalized)
    if normalized[0].kind_of? Vector
      arr = normalized[0].to_a
      # get the "transpose" - a list of component lists
      components = arr.zip(*normalized[1..normalized.length])
      # find the sdev of each component and pack it up into a Vector
      Vector[*components.map(&method(:sdev))]
    else
      sum = 0
      normalized.each do |x|
        sum += x**2
      end
      Math.sqrt(sum / normalized.size)
    end
  end

  # Result of the same preprocessing on the training set with the given input.
  # Params:
  # +x+:: The input value to use.
  def pack(x)
    out = x
    @history.each do |event|
      case event[0]
      when :normalize_classes
      when :normalize_mean_x
        out = pack_mean(out, event[1])
      when :normalize_mean_y
      when :scale_x
        out = pack_scale(out, event[1])
      when :scale_y
      end
    end
    out
  end

  # Undoes the processing on the given output.
  # Params:
  # +y+:: The processed output value to use.
  def unpack(y)
    out = y
    @history.reverse.each do |event|
      case event[0]
      when :normalize_classes
        out = event[1][out]
      when :normalize_mean_x
      when :normalize_mean_y
        out = unpack_mean(out, event[1])
      when :scale_x
      when :scale_y
        out = unpack_scale(out, event[1])
      end
    end
    out
  end
end

if __FILE__ == $0
  puts "* Testing preprocessor..."
  puts "1 Standardizing regression scenario."
  puts "1 Input: Vector[10 * i + 1000, i**2 - 3000] for 0 <= i < 10"
  puts "1 Output: Vector[-i + 19.5] for 0 <= i < 10"
  xs = Array.new(10) { |i| Vector[10 * i + 1000, i**2 - 3000] }
  ys = Array.new(10) { |i| Vector[-i + 19.5] }
  pp = Preprocessor.new(xs, ys)
  pp.standardize
  puts "! Standardized input:"
  puts "  #{pp.xs}"
  puts "! Standardized output:"
  puts "  #{pp.ys}"
  puts "! Log:"
  puts "  #{pp.history}"
  puts "! Using the same input scheme for Vector[1000, -3000]:"
  puts "  #{pp.pack(Vector[1000, -3000])}"
  puts "! Reversing the scheme for output Vector[1.5666989036012806]:"
  puts "  #{pp.unpack(Vector[1.5666989036012806])}"
  xs2 = [1, 2, 3, 4, 5, 6, 7, 8]
  ys2 = ["oranges", "apples", "pears", "bananas", "bananas", "pears", "apples", "oranges"]
  puts "2 Standardizing/normalizing classification scenario."
  puts "2 Input: #{xs2}"
  puts "2 Output: #{ys2}"
  pp2 = Preprocessor.new(xs2, ys2)
  pp2.standardize(regression=false)
  puts "! Standardized input:"
  puts "  #{pp2.xs}"
  puts "! Standardized output:"
  puts "  #{pp2.ys}"
  puts "! Log:"
  puts "  #{pp2.history}"
  puts "! Using the same input scheme for 3:"
  puts "  #{pp2.pack(3)}"
  puts "! Reversing the scheme for output 3:"
  puts "  #{pp2.unpack(3)}"

end