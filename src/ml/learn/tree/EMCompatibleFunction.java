package ml.learn.tree;

/**
 * Defining an interface for functions to be used by {@link EMAlgo}
 * @see EMAlgo
 * @author Aldrian Obaja <aldrianobaja.m@gmail.com>
 *
 */
public interface EMCompatibleFunction {
	/**
	 * Calculate the expected counts of the features given the weights
	 * @param weights
	 * @return
	 */
	public double[] expectation(double[] weights);
	
	/**
	 * Maximize the expectation by adjusting the weights
	 * @param expectations
	 * @return
	 */
	public double[] maximize(double[] expectations);
	
	/**
	 * Returns the number of parameters defined for this function
	 * @return
	 */
	public int numParams();
}
