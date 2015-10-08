package ml.learn.tree;

public interface EMCompatibleFunction {
	public double[] expectation(double[] weights);
	public double[] maximize(double[] expectations);
	public int numParams();
}
