import java.util.ArrayList;
import java.util.Random;

public class NeuralNet extends SupervisedLearner {
  protected Vec weights;
  protected Vec gradient;
  protected ArrayList<Layer> layers;
  protected Random random;

  String name() { return ""; }

  NeuralNet(Random r) {
    layers = new ArrayList<Layer>();
    random = r;
  }

  void initWeights() {
    // Calculate the total number of weights
    int weightsSize = 0;
    for(int i = 0; i < layers.size(); ++i) {
      Layer l = layers.get(i);
      weightsSize += l.getNumberWeights();
    }
    weights = new Vec(weightsSize);
    gradient = new Vec(weightsSize);

    int pos = 0;
    for(int i = 0; i < layers.size(); ++i) {
      Layer l = layers.get(i);

      int weightsChunk = l.getNumberWeights();
      Vec w = new Vec(weights, pos, weightsChunk);

      l.initWeights(w, random);

      pos += weightsChunk;
    }
  }

  void backProp(Vec target) {
    Vec blame = new Vec(target.size());
    blame.add(target);
    blame.addScaled(-1, layers.get(layers.size()-1).activation);


    int pos = weights.size();
    for(int i = layers.size()-1; i >= 0; --i) {
      Layer l = layers.get(i);
      //l.debug();
      //prevBlame = new Vec(l.inputs);

      int weightsChunk = l.getNumberWeights();
      pos -= weightsChunk;
      Vec w = new Vec(weights, pos, weightsChunk);

      blame = l.backProp(w, blame);
    }
  }

  void updateGradient(Vec x) {

    int pos = 0;
    for(int i = 0; i < layers.size(); ++i) {
      Layer l = layers.get(i);
      int gradChunk = l.getNumberWeights();
      Vec v = new Vec(gradient, pos, gradChunk);

      l.updateGradient(x, v);
      x = new Vec(l.activation);
      pos += gradChunk;
    }
  }

  // void refineWeights(Vec x, Vec y, Vec weights, double learning_rate, Training type) {
  //
  //   if(Training.STOCHASTIC == type) {
  //     gradient.fill(0.0);
  //   } else if(Training.MOMENTUM == type) {
  //     gradient.scale(0.9);
  //   }
  //
  //   predict(x);
  //
  //   // Compute the blame on each layer
  //   backProp(y);
  //
  //   // Compute the gradient
  //   updateGradient(x);
  //
  //   // Adjust the weights per the learning_rate
  //   this.weights.addScaled(learning_rate, gradient);
  // }

  void refineWeights(double learning_rate) {
    weights.addScaled(learning_rate, gradient);
  }

  Vec predict(Vec in) {
    int pos = 0;
    for(int i = 0; i < layers.size(); ++i) {
      Layer l = layers.get(i);
      int weightsChunk = l.getNumberWeights();
      Vec v = new Vec(weights, pos, weightsChunk);
      l.activate(v, in);
      in = l.activation;
      pos += weightsChunk;
    }

    return (layers.get(layers.size()-1).activation);
  }

  void train(Matrix features, Matrix labels, int[] indices, int batch_size, double momentum) {
    if(batch_size < 1)
      throw new IllegalArgumentException("Batch Size is invalid!");

    // get the set of indices
    // int[] trainingIndices = new int[features.rows()];

    Vec in, target;
    for(int i = 0; i < features.rows(); ++i) {
      in = features.row(indices[i]);
      target = labels.row(indices[i]);

      predict(in);
      backProp(target);
      updateGradient(in);

      if(i % batch_size == 0) {
        double scale_learning = (1.0 / i + 1.0);
        refineWeights(0.0175 * i);
        if(momentum <= 0)
          gradient.fill(0.0);
        else
          gradient.scale(momentum);
      }
    }
  }

  /// Train this supervised learner
  // void train(Matrix features, Matrix labels, Training type) {
  //   int batch_size = 10;
  //   int[] trainingIndices =  new int[features.rows()];
  //   // double[] testIndices = new double[]
  //
  //   if(Training.STOCHASTIC == type) {
  //     /// Update weights every training pattern
  //
  //     Vec in, target;
  //     for(int i = 0; i < features.rows(); ++i) {
  //       gradient.fill(0.0);
  //       in = features.row(trainingIndices[i]);
  //
  //       target = new Vec(10);
  //       target.vals[(int) labels.row(trainingIndices[i]).get(0)] = 1;
  //
  //       predict(in);
  //       backProp(target);
  //       updateGradient(in);
  //       refineWeights(0.0175);
  //     }
  //
  //     scrambleIndices(random, trainingIndices, null);
  //   } else if(Training.BATCH == type) {
  //     /// Update weights after the entire training set
  //
  //     Vec in, target;
  //     for(int i = 0; i < features.rows(); ++i) {
  //       in = features.row(trainingIndices[i]);
  //
  //       target = new Vec(10);
  //       target.vals[(int) labels.row(trainingIndices[i]).get(0)] = 1;
  //
  //       predict(in);
  //       backProp(target);
  //       updateGradient(in);
  //     }
  //
  //     refineWeights(0.0175);
  //     gradient.fill(0.0);
  //     scrambleIndices(random, trainingIndices, null);
  //   } else if(Training.MINIBATCH == type) {
  //     /// Update weights after batch_size of patterns
  //
  //     Vec in, target;
  //     for(int i = 0; i < features.rows(); ++i) {
  //       in = features.row(trainingIndices[i]);
  //
  //       target = new Vec(10);
  //       target.vals[(int) labels.row(trainingIndices[i]).get(0)] = 1;
  //
  //       predict(in);
  //       backProp(target);
  //       updateGradient(in);
  //
  //       if(i % batch_size == 0) {
  //         refineWeights(0.0175);
  //         gradient.fill(0.0);
  //         scrambleIndices(random, trainingIndices, null);
  //       }
  //     }
  //
  //     refineWeights(0.0175);
  //     gradient.fill(0.0);
  //     scrambleIndices(random, trainingIndices, null);
  //   } else if(Training.MOMENTUM == type) {
  //     /// Update weights after every pattern, scaling gradient by 0.9
  //
  //     Vec in, target;
  //     gradient.fill(0.0);
  //     for(int i = 0; i < features.rows(); ++i) {
  //       in = features.row(trainingIndices[i]);
  //
  //       target = new Vec(10);
  //       target.vals[(int) labels.row(trainingIndices[i]).get(0)] = 1;
  //
  //       predict(in);
  //       backProp(target);
  //       updateGradient(in);
  //       refineWeights(0.0175);
  //       gradient.scale(0.9);
  //     }
  //   } else {
  //     throw new IllegalArgumentException("No usable training method given: " + type);
  //   }
  //
  //
  // }

}
