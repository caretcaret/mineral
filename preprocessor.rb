#!/usr/bin/ruby
# coding: utf-8

require 'matrix'

class Preprocessor
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
  end

  def standardize(xs_only=false)
    mean_x = mean(@xs)
    # mean normalization
    @xs.map! { |x| x - mean_x }

    # feature scaling
    sdev_x = sdev(@xs)
    @xs.map! do |x|
      x.map_with_index do |x_i, i|
        if sdev_x[i] == 0
          # if standard deviation is 0, then the normalized features
          # must all be 0. Just set it to 0
          0
        else
          # otherwise divide by sdev to scale the feature
          x_i / sdev_x[i]
        end
      end
    end

    if !@ys.empty? && !xs_only
      mean_y = mean(@ys)
      @ys.map! { |y| y - mean_y }

      sdev_y = sdev(@ys)
      @ys.map! do |y|
        y.map_with_index do |y_i, i|
          if sdev_y[i] == 0
            0
          else
            y_i / sdev_y[i]
          end
        end
      end

      @history << [:standardize_x, mean_x, sdev_x]
      @history << [:standardize_y, mean_y, sdev_y]
    else
      @history << [:standardize_x, mean_x, sdev_x]
    end
  end

  def mean(list)
    if list[0].kind_of? Vector
      sum = Vector[*Array.new(list[0].size) { 0 }]
    else
      sum = 0
    end

    1.0 / list.size * sum
  end

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
      Math.sqrt(sum / list.size)
    end
  end

  def pack(x)

  end

  def unpack(y)

  end
end

if __FILE__ == $0

end