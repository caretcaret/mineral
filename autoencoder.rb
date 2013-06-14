#!/usr/bin/ruby
# coding: utf-8

require './neural_network'
require 'matrix'

class Autoencoder < NeuralNetwork

  # Creates a stacked autoencoder including the decoding layers.
  # Params:
  # +layers+:: An array of the number of neurons in each layer, from
  # lowest to highest level (whose dimension is equal to the desired encoding).
  # +xs+:: The training examples as an array of vectors.
  # +norm_weight+:: The normalization weight to use in the neural network.
  # +weights+:: The weights of the autoencoder.
  # +trained+:: The number of layers trained so far.
  def initialize(layers, xs, norm_weight=0, weights=nil, trained=0)
    @trained = trained
    @layers = layers
    raise "No layers" unless @layers.size > 0
    # Reflect the layers of the autoencoder
    full_layers = layers + layers.reverse[1...layers.size]
    super(full_layers, xs, xs, norm_weight, weights)
  end

  def pretrain_layer(layer, rate, monitor=nil, halt=nil)
    raise "No more layers to train" unless @trained < @layers.size
    # The layer to train, going deeper into the encoder
    deeper = layer
    # The other layer to train, going up from the encoder
    shallower = 2 * @layers.size - 1 - deeper
    deeper_matrix = @weights[deeper]
    shallower_matrix = @weights[shallower]
    features = encoded_xs(deeper)
    ann = NeuralNetwork.new([@layers[deeper]], features, features, @norm_weight, [deeper_matrix, shallower_matrix])
    ann.train(rate, monitor, halt)
    @weights[deeper] = ann.weights[0]
    @weights[shallower] = ann.weights[1]
  end

  def pretrain(rate, monitor=nil, halt=nil)
    (@trained...@layers.size).each do |i|
      pretrain_layer(i, rate, monitor, halt)
      @trained += 1
    end
  end

  def encode(x, depth=@layers.size)
    if depth == 0
      x
    else
      forward!(x)
      remove_bias_elem(@activations[depth])
    end
  end

  def encoded_xs(depth=@layers.size)
    @xs.map { |x| encode(x, depth) }
  end
end

if __FILE__ == $0
  puts "* Testing autoencoder..."
  xs = (0..10).map do |x|
    Vector[-x / 10.0 + 0.5, x / 10.0 - 0.5]
  end
  puts "1 Input:"
  puts "1 #{xs}"
  sae = Autoencoder.new([1], xs, 0)
  monitor = lambda { |gd|
    if gd.iterations % 20 == 0
      print " #{gd.iterations}"
    end
  }
  puts "  Pretraining..."
  sae.pretrain(1, monitor)
  puts "\n  Training..."
  sae.train(1, monitor)

  puts "\n! Weights:"
  puts "  #{sae.weights}"
  puts "! Reconstructed input:"
  puts "  #{xs.map { |x| sae.evaluate!(x) }}"
  puts "! Encoding:"
  puts "  #{sae.encoded_xs}"
end