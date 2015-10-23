package ml.learn.tree;

/**
 * A class implementing EM algorithm
 * @author Aldrian Obaja <aldrianobaja.m@gmail.com>
 *
 */
public class EMAlgo {
	
	public EMCompatibleFunction function;
	public int maxIter;
	public double threshold;
	
	public static final int DEFAULT_MAX_ITER = 1000;
	public static final double DEFAULT_THRESHOLD = 1e-6;
	
	public EMAlgo(EMCompatibleFunction function){
		this(function, DEFAULT_MAX_ITER, DEFAULT_THRESHOLD);
	}
	
	public EMAlgo(EMCompatibleFunction function, int maxIter, double threshold){
		this.function = function;
		this.maxIter = maxIter;
		this.threshold = threshold;
	}
	
	/**
	 * Find the best parameters using the EM algorithm on the function specified during the construction
	 * @param startingPoint
	 * @return
	 */
	public double[] getBestParams(){
		double[] startingPoint = new double[function.numParams()];
		for(int i=0; i<startingPoint.length; i++){
			startingPoint[i] = 1.0;
		};
		startingPoint = function.maximize(startingPoint);
		System.out.println("Starting point:");
		for(int i=0; i<startingPoint.length; i++){
			System.out.printf("%.3f ", startingPoint[i]);
		};
		System.out.println();
		double[] prevResult = startingPoint;
		double[] result = null;
		double[] expectations = null;
		long start = System.currentTimeMillis();
//		long startTime, endTime;
		for(int iterNum=0; iterNum<maxIter; iterNum++){
//			System.out.println("Starting iteration "+(iterNum+1));
//			startTime = System.currentTimeMillis();
			expectations = function.expectation(prevResult);
//			endTime = System.currentTimeMillis();
//			System.out.printf("Finished calculating expectation in %.3fs\n", (endTime-startTime)/1000.0);
//			startTime = System.currentTimeMillis();
			result = function.maximize(expectations);
//			endTime = System.currentTimeMillis();
//			System.out.printf("Finished maximizing params in %.3fs\n", (endTime-startTime)/1000.0);
			double squaredChange = 0.0;
			for(int i=0; i<result.length; i++){
				squaredChange += Math.pow(result[i]-prevResult[i], 2);
			}
			squaredChange /= result.length;
			System.out.printf("Iteration %d: Average squared change: %.7f, elapsed time: %.3fs\n", iterNum+1, squaredChange, (System.currentTimeMillis()-start)/1000.0);
			if(squaredChange < threshold){
				break;
			}
			for(int i=0; i<result.length; i++){
				prevResult[i] = result[i];
			}
		}
		function.setParams(result);
		return result;
	}
}
