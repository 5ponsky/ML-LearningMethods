// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------
import java.util.Random;

abstract class SupervisedLearner
{
	/// Return the name of this learner
	abstract String name();

	/// Train this supervised learner
	abstract void train(Matrix features, Matrix labels, int batch_size, double momentum);

	/// Make a prediction
	abstract Vec predict(Vec in);

	/// If the data patterns are merged with the labels seperate them
	// Assumes labels are vectors
	void splitLabels(Matrix data, Matrix features, Matrix labels) {
		// copy the features over
		features.setSize(data.rows(), data.cols()-1);
		features.copyBlock(0, 0, data, 0, 0, data.rows(), data.cols()-1);

		// Copies the labels over
		// This assumes labels are a vector
		labels.setSize(data.rows(), 1);
		labels.copyBlock(0, 0, data, 0, data.rows()-1, data.rows(), 1);
	}

	/// Splits into training/testing, training = total * splitRatio
	// Assumes that the labels are a vector
	void splitData(Matrix featureData, Matrix labelData, Matrix trainingFeatures, Matrix trainingLabels,
		Matrix testingFeatures, Matrix testingLabels, double splitRatio) {

		int trainingSize = (int)(featureData.rows() * splitRatio);

		// copy the training set
		trainingFeatures.setSize(trainingSize, featureData.cols());
		trainingLabels.setSize(trainingSize, labelData.cols());

		trainingFeatures.copyBlock(0, 0, featureData, 0, 0, trainingSize, featureData.cols());
		trainingLabels.copyBlock(0, 0, labelData, 0, 0, trainingSize, labelData.cols());

		// for(int i = 0; i < trainingSize; ++i) {
		// 	for(int j = 0; j < featureData.cols(); ++j) {
		// 		double newEntry = featureData.row(i).get(j);
		// 		trainingFeatures.row(i).set(j, newEntry);
		// 	}
		//
		// 	for(int j = 0; j < labelData.cols(); ++j) {
		// 		double newEntry = labelData.row(i).get(j);
		// 		trainingLabels.row(i).set(j, newEntry);
		// 	}
		// }

		// copy the test set
		testingFeatures.setSize(featureData.rows() - trainingSize, featureData.cols());
		testingLabels.setSize(labelData.rows() - trainingSize, labelData.cols());

		testingFeatures.copyBlock(0, 0, featureData,
			trainingSize, 0, featureData.rows()-trainingSize, featureData.cols());
		testingLabels.copyBlock(0, 0, labelData,
			trainingSize, 0, labelData.rows()-trainingSize, labelData.cols());

		// for(int i = trainingSize; i < featureData.rows(); ++i) {
		// 	int i_adjusted = i - trainingSize;
		// 	for(int j = 0; j < featureData.cols(); ++j) {
		// 		double newEntry = featureData.row(i).get(j);
		// 		testingFeatures.row(i_adjusted).set(j, newEntry); // fix i
		// 	}
		//
		// 	for(int j = 0; j < labelData.cols(); ++j) {
		// 		double newEntry = labelData.row(i).get(j);
		// 		testingLabels.row(i_adjusted).set(j, newEntry); // fix i
		// 	}
		// }

	}


	void convergence() {

	}

	/// Measures the misclassifications with the provided test data
	int countMisclassifications(Matrix features, Matrix labels) {
		if(features.rows() != labels.rows())
			throw new IllegalArgumentException("Mismatching number of rows");
		int mis = 0;
		for(int i = 0; i < features.rows(); i++) {
			Vec feat = features.row(i);
			Vec pred = predict(feat);
			Vec lab = formatLabel((int)labels.row(i).get(0));
			if(poorClassification(pred, lab)) {
				mis++;
			}
		}
		return mis;
	}

	Vec formatLabel(int label) {
		if(label > 9 || label < 0)
			throw new IllegalArgumentException("not a valid labels!");

		double[] res = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
		res[label] = 1;
		return new Vec(res);
	}

	boolean poorClassification(Vec pred, Vec lab) {
		if(pred.size() != lab.size())
			throw new IllegalArgumentException("vector size mismatch!");

		pred.oneHot();
		for(int i = 0; i < pred.size(); ++i) {
			if(pred.get(i) != lab.get(i))
				return true;
		}
		return false;
	}


	double sum_squared_error(Matrix features, Matrix labels) {
		if(features.rows() != labels.rows())
			throw new IllegalArgumentException("Mistmatching number of rows");

		double mis = 0;
		for(int i = 0; i < features.rows(); i++) {
			Vec feat = features.row(i);
			Vec pred = predict(feat);
			Vec lab = labels.row(i);
			for(int j = 0; j < lab.size(); j++) {
				double blame = (lab.get(j) - pred.get(j)) * (lab.get(j) - pred.get(j));
				System.out.println(i + " " + pred);
				mis = mis + blame;
			}
		}

		return mis;
	}

	double cross_validation(int r, int f, Matrix featureData, Matrix labelData) {
		Random random = new Random(1234);

		// Cross-Validation indices
		int repititions = r;
		int folds = f;
		double foldRatio = 1.0 / (double)folds;
		int beginStep = 0;
		int endStep = 1;
		int testBlockSize = (int)(featureData.rows() * foldRatio);
		int beginIndex = 0;
		int endIndex = 0;

		// Create train matrices
		Matrix trainFeatures = new Matrix((int)(featureData.rows() - Math.floor(featureData.rows()*foldRatio)), featureData.cols());
		Matrix trainLabels = new Matrix((int)(featureData.rows() - Math.floor(featureData.rows()*foldRatio)), labelData.cols());

		// Create test matrices
		Matrix testFeatures = new Matrix((int)(featureData.rows()*foldRatio), featureData.cols());
		Matrix testLabels = new Matrix((int)(featureData.rows()*foldRatio), labelData.cols());


		// Partition the data by folds
		double sse = 0; // Sum squared error
		double mse = 0; // Mean squared error
		double rmse = 0; // Root mean squared error


		for(int k = 0; k < repititions; ++k) {
			for(beginStep = 0; beginStep < folds; ++beginStep) {
				beginIndex = beginStep * (featureData.rows() / folds);
				endIndex = (beginStep + 1) * (featureData.rows() / folds);

				// First Training block
				trainFeatures.copyBlock(0, 0, featureData, 0, 0, beginIndex, featureData.cols());
				trainLabels.copyBlock(0, 0, labelData, 0, 0, beginIndex, labelData.cols());


				// Test block
				testFeatures.copyBlock(0, 0, featureData, beginIndex, 0, endIndex-beginIndex, featureData.cols());
				testLabels.copyBlock(0, 0, labelData, beginIndex, 0, endIndex-beginIndex, labelData.cols());


				// 2nd Training block
				trainFeatures.copyBlock(beginIndex, 0, featureData,
					beginIndex+1, 0, featureData.rows() - endIndex, featureData.cols());
				trainLabels.copyBlock(beginIndex, 0, labelData,
					beginIndex+1, 0, featureData.rows() - endIndex, labelData.cols());


				train(trainFeatures, trainLabels, 1, 0.0);
				sse = sse + sum_squared_error(testFeatures, testLabels);
			}

			mse = mse + (sse / featureData.rows());
			sse = 0;

			for(int i = 0; i < featureData.rows(); ++i) {
				int selectedRow = random.nextInt(featureData.rows());
				int destinationRow = random.nextInt(featureData.rows());
				featureData.swapRows(selectedRow, destinationRow);
				labelData.swapRows(selectedRow, destinationRow);
			}
		}


		rmse = Math.sqrt(mse/repititions);
		return rmse;
	}

	void scrambleIndices(Random random, int[] trainingIndices, int[] testIndices) {
		for(int i = 0; i < trainingIndices.length * 0.5; ++i) {
			int randomIndex = random.nextInt(trainingIndices.length);
			int temp = trainingIndices[i];
			trainingIndices[i] = trainingIndices[randomIndex];
			trainingIndices[randomIndex] = temp;

		}

		if(testIndices != null) {
			for(int i = 0; i < testIndices.length * 0.5; ++i) {
				int randomIndex = random.nextInt(testIndices.length);
				int temp = testIndices[i];
				testIndices[i] = testIndices[randomIndex];
				testIndices[randomIndex] = temp;
			}
		}
	}
}
