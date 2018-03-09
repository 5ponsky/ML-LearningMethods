


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
  void train(Matrix features, Matrix labels, Training type) {
    // Replace missing data entries with centroid
    Matrix imputed = process(imputer, features);
    // Normalize data
    Matrix normalized = process(normalizer, imputed);
    // replace categorical variables with one-hot continuous arrays
    Matrix nomcated = process(nomcat, normalized);

    // int mis = 0;
    // while(mis < 350) {
    //   mis = nn.countMisclassifications(nomcated, labels);
    // }

  }

  Matrix process(PreprocessingOperation po, Matrix data) {
    Matrix output = po.outputTemplate();
    for(int i = 0; i < data.rows(); ++i) {
      double[] in = data.row(i).vals();
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
