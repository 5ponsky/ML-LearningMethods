


public class Filter extends SupervisedLearner {
  NeuralNet nn;
  Filter filter;
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

  String name() { return ""; }

  /// Uses nomcat to convert categorical variables into values for NeuralNet
  void train(Matrix features, Matrix labels, Training type) {


  }

  Vec predict(Vec in) {
    //nomcat.transform
    return new Vec(1);
  }


}
