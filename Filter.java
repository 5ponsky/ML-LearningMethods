


public class Filter extends SupervisedLearner {
  NeuralNet nn;
  Filter filter;
  Imputer imputer;
  NomCat nomcat;
  Normalizer normalizer;

  Filter() {}

  Filter(Filter f) {
    filter = f;
    normalizer = new Normalizer();
  }

  Filter(NeuralNet nn) {
    this.nn = nn;
    nomcat = new NomCat();
  }

  Filter(Imputer im, Normalizer n, NomCat nm, NeuralNet nnt) {
    imputer = im;
    normalizer = n;
    nomcat = nm;
    nn = nnt;
  }

  String name() { return ""; }

  /// process data into a format readily available for training
  void train(Matrix features, Matrix labels, int batch_size, double momentum) {
    // Replace missing data entries with centroid
    //System.out.println("F: " + features);
    Matrix imputed = process(imputer, features);
    //System.out.println("i: " + imputed);
    // Normalize data
    Matrix normalized = process(normalizer, imputed);
    //System.out.println("n: " + normalized);
    // replace categorical variables with one-hot continuous arrays
    Matrix nomcated = process(nomcat, normalized);
    System.out.println("nom: " + nomcated);
    //System.out.println("-----------------");

    /// I want some intelligent way of getting the input and outputs
    nn.layers.add(new LayerLinear(nomcated.cols(), 100));
    nn.layers.add(new LayerTanh(100));

    // nn.layers.add(new LayerLinear(80, 30));
    // nn.layers.add(new LayerTanh(30));

    nn.layers.add(new LayerLinear(100, 4));
    nn.layers.add(new LayerTanh(4));

    nn.initWeights();

    // Train the neural net
    nn.train(nomcated, labels, batch_size, momentum);
  }

  /// produces a matrix of processed data under the operation
  Matrix process(PreprocessingOperation po, Matrix data) {
    po.train(data);
    Matrix output = po.outputTemplate();
    for(int i = 0; i < data.rows(); ++i) {
      double[] in = data.row(i).vals;
      double[] out = new double[data.cols()];
      po.transform(in, out);
      output.takeRow(out);
    }
    return output;
  }

  Vec predict(Vec in) {
    //nomcat.transform
    return new Vec(1);
  }


}
