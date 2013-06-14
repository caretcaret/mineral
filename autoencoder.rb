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
    shallower = 2 * @layers.size - 1 - layer
    deeper_matrix = @weights[deeper]
    shallower_matrix = @weights[shallower]
    features = encoded_xs(deeper)
    ann = NeuralNetwork.new([@layers[deeper]], features, features, @norm_weight, [deeper_matrix, shallower_matrix])
    ann.train(rate, monitor, halt)
    @weights[deeper] = ann.weights[0]
    @weights[shallower] = ann.weights[1]
  end

  def pretrain(rate, monitor=nil, halt=nil)
    @layers.each_index do |i|
      pretrain_layer(i, rate, monitor, halt)
    end
  end

  def encode(x, depth)
    if depth == 0
      x
    else
      forward!(x)
      remove_bias_element(@activations[depth])
    end
  end

  def encoded_xs(depth)
    @xs.map { |x| encode(x, depth) }
  end
end
