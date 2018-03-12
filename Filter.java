import java.util.Random;


public class Filter extends SupervisedLearner {
  NeuralNet nn;
  Filter filter;
  Random random;

  Filter() {}

  Filter(NeuralNet nn, Random r) {
    this.nn = nn;
    random = r;
  }

  String name() { return ""; }

  /// process data into a format readily available for training
  void train(Matrix features, Matrix labels, int[] indices, int batch_size, double momentum) {
    System.out.println("Labels pre: " + labels.cols());
    features.copy(preProcess(features, new Imputer(),
        new Normalizer(), new NomCat()));
    labels.copy(preProcess(labels, new Imputer(),
        new Normalizer(), new NomCat()));
    System.out.println("Labels post: " + labels.cols());
  }

  /// Train the NeuralNet
  void trainNeuralNet(Matrix features, Matrix labels, int[] indices, int batch_size,
    double momentum) {
    nn.train(features, labels, indices, batch_size, momentum);
    scrambleIndices(random, indices, null);
  }

  Matrix preProcess(Matrix data, Imputer im, Normalizer norm, NomCat nm) {
    Matrix imputed = transform(im, data);
    Matrix normalized = transform(norm, imputed);
    Matrix nomcated = transform(nm, normalized);
    return nomcated;
  }

  /// produces a matrix of transformed data under the operation
  Matrix transform(PreprocessingOperation po, Matrix data) {
    po.train(data);
    Matrix output = po.outputTemplate();
    for(int i = 0; i < data.rows(); ++i) {
      double[] in = data.row(i).vals;
      double[] out = new double[output.cols()];
      po.transform(in, out);
      output.takeRow(out);
    }
    return output;
  }

  /// Untransformation process
  Vec postProcess(Vec prediction, NomCat nm, Normalizer norm, Imputer im) {
    Vec un_nomcated = untransform(nm, prediction);
    Vec un_normalized = untransform(norm, un_nomcated);
    Vec un_imputed = untransform(im, un_normalized);
    return un_imputed;
  }

  /// Transforms a vector back into its nominal representation
  Vec untransform(PreprocessingOperation po, Vec prediction) {
    double[] output = new double[1]; /// I HATE THIS HARDCODED VALUE
    po.untransform(prediction.vals, output);
    return new Vec(output);
  }

  /// Overloaded superclass definition for filtered data
  int countMisclassifications(Matrix features, Matrix labels) {
    if(features.rows() != labels.rows())
			throw new IllegalArgumentException("Mismatching number of rows");

    // PreProcess the features
    Matrix processedTestFeatures = preProcess(features, new Imputer(),
      new Normalizer(), new NomCat());

    // So in this function we actually need a persistent set of preprocessors
    // That are trained on labels so we can re-transform them back into
    // their nominal representation for misclassification
    Imputer im = new Imputer();
    Normalizer norm = new Normalizer();
    NomCat nm = new NomCat();
    Matrix processedTestLabels = preProcess(labels, im, norm, nm);

    // predict a value and check if it is an accurate prediction
		int mis = 0;
		for(int i = 0; i < processedTestFeatures.rows(); i++) {
			Vec feat = processedTestFeatures.row(i);
      Vec lab = labels.row(i);

      Vec prediction = nn.predict(feat);
      Vec out = postProcess(prediction, nm, norm, im);

      // Component-wise comparison of nominal values
      for(int j = 0; j < lab.size(); ++j) {
        if(out.get(j) != lab.get(j))
					mis++;
      }

		}
		return mis;
  }

  Vec predict(Vec in) {
    throw new RuntimeException("This class does not use predict!");
  }

}
