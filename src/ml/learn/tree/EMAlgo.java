package ml.learn.tree;

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
	
	public double[] getBestParams(double[] startingPoint){
		double[] prevResult = startingPoint;
		double[] result = null;
		double[] expectations = null;
		long startTime, endTime;
		for(int iterNum=0; iterNum<maxIter; iterNum++){
			System.out.println("Starting iteration "+(iterNum+1));
			startTime = System.currentTimeMillis();
			expectations = function.expectation(prevResult);
			endTime = System.currentTimeMillis();
			System.out.printf("Finished calculating expectation in %.3fs\n", (endTime-startTime)/1000.0);
			startTime = System.currentTimeMillis();
			result = function.maximize(expectations);
			endTime = System.currentTimeMillis();
			System.out.printf("Finished maximizing params in %.3fs\n", (endTime-startTime)/1000.0);
			boolean hasLargeChange = false;
			for(int i=0; i<result.length; i++){
				if(Math.abs(result[i]-prevResult[i]) >= threshold){
					hasLargeChange = true;
					break;
				}
			}
			if(!hasLargeChange){
				break;
			}
			for(int i=0; i<result.length; i++){
				prevResult[i] = result[i];
			}
		}
		return result;
	}
}
