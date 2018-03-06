


public class Filter extends SupervisedLearner {
  NeuralNet nn;
  Filter filter;


  Filter() {
    //nn = new NeuralNet();
  }

  Filter(Filter f) {
    filter = f;
  }

  String name() { return ""; }

  void train(Matrix features, Matrix labels, Training type) {

  }

  Vec predict(Vec in) {
    return new Vec(1);
  }


}
