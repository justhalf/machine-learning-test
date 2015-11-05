package ml.learn.linear;

import java.util.List;
import java.util.Map;

import ml.learn.object.Tag;

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
	
	/**
	 * Return the list of tags
	 * @return
	 */
	public Map<Tag, Integer> getTags();
}
