package ml.learn;

import java.util.List;

public interface StructuredClassifier {

	/**
	 * Train the classifier
	 * @param trainingData
	 */
	public void train(List<Instance> trainingData);
	
	/**
	 * Predict the sequence of tags of the given test data
	 * @param testData
	 * @return
	 */
	public List<Instance> predict(List<Instance> testData);
}
