import java.util.ArrayList;
import java.util.Random;

public class NeuralNet extends SupervisedLearner {
  protected Vec weights;
  protected Vec gradient;
  protected ArrayList<Layer> layers;

  // This is a temporary architecture decision;
  // I need to be able to pause the training at any moment to test the net,
  // But resume exactly where I left off.  I didn't choose to pass as a parameter,
  // Because Idk if I want to change every supervised learner yet.
  public int trainingProgress;


  String name() { return ""; }

  NeuralNet(Random r) {
    super(r);
    layers = new ArrayList<Layer>();

    trainingProgress = 0;
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

      l.initWeights(w, this.random);

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

    // How many patterns/mini-batches should we train on before testing?
    int cutoff = 1;

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

        if(i % cutoff == 0) {
          trainingProgress = i;
          break;
        }
      }
    }
  }

}
