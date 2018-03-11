


public class Filter extends SupervisedLearner {
  NeuralNet nn;
  Filter filter;

  Filter() {}

  Filter(Filter f) {
  }

  Filter(NeuralNet nn) {
    this.nn = nn;
  }

  Filter(Imputer im, Normalizer n, NomCat nm, NeuralNet nnt) {
    nn = nnt;
  }

  String name() { return ""; }

  /// process data into a format readily available for training
  void train(Matrix features, Matrix labels, int batch_size, double momentum) {
    // // Replace missing data entries with centroid
    // Matrix imputedFeatures = transform(imputer, features);
    // Matrix imputedLabels = transform(imputer, labels);
    // // Normalize data
    // Matrix normalizedFeatures = transform(normalizer, imputedFeatures);
    // Matrix normalizedLabels = transform(normalizer, imputedLabels);
    // // replace categorical variables with one-hot continuous arrays
    // Matrix nomcatedFeatures = transform(nomcat, normalizedFeatures);
    // Matrix nomcatedLabels = transform(nomcat, normalizedLabels);

    Matrix processedFeatures = preProcess(features);
    Matrix processedLabels = preProcess(labels);

    /// I want some intelligent way of getting the input and outputs
    nn.layers.add(new LayerLinear(processedFeatures.cols(), 100));
    nn.layers.add(new LayerTanh(100));

    // nn.layers.add(new LayerLinear(80, 30));
    // nn.layers.add(new LayerTanh(30));

    nn.layers.add(new LayerLinear(100, 4));
    nn.layers.add(new LayerTanh(4));

    nn.initWeights();

    // Train the neural net
    nn.train(processedFeatures, processedLabels, batch_size, momentum);


  }

  Matrix preProcess(Matrix data) {
    Matrix imputed = transform(new Imputer(), data);
    Matrix normalized = transform(new Normalizer(), imputed);
    Matrix nomcated = transform(new NomCat(), normalized);
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

  Vec untransform(PreprocessingOperation po, Vec prediction) {
    double[] output = new double[po.outputTemplate().cols()];
    po.untransform(prediction.vals, output);
    return new Vec(output);
  }

  /// Overloaded predict
  Vec predict(Vec in, Imputer im, Normalizer norm, NomCat nm) {
    Vec imputed = transform(im, in);
    Vec normalized = transform(norm, imputed);
    Vec nomcated = transform(nm, normalized);
    return nomcated;
  }

  /// Produces a Vec of transformed data (this is used for testing)
  Vec transform(PreprocessingOperation po, Vec in) {

    Vec output = new Vec(po.outputTemplate().cols());
    po.transform(in.vals, output.vals);
    return output;
  }

  /// Overloaded superclass definition for filtered data
  int countMisclassifications(Matrix features, Matrix labels) {
    if(features.rows() != labels.rows())
			throw new IllegalArgumentException("Mismatching number of rows");

		int mis = 0;
    Imputer im = new Imputer();
    Normalizer norm = new Normalizer();
    NomCat nm = new NomCat();
		for(int i = 0; i < features.rows(); i++) {
			Vec feat = features.row(i);
      System.out.println(feat.size());
			Vec transformed = predict(feat, im, norm, nm);
      Vec out = postProcess(transformed, nm, norm, im);
      Vec lab = labels.row(i);

      for(int j = 0; j < lab.size(); ++j) {
        if(out.get(j) != lab.get(j))
					mis++;
      }
		}
		return mis;
  }

  Vec predict(Vec in) {
      throw new RuntimeException("Filter does not use regular predict() method!");
  }
}
