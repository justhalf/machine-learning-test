package ml.learn.util;

public class Common {

	public static class AccumulatorResult{
		public double value;
		public int maxIdx;
		
		public AccumulatorResult(double value, int maxIdx){
			this.value = value;
			this.maxIdx = maxIdx;
		}
		
		public AccumulatorResult(double value){
			this(value, -1);
		}
	}

	/**
	 * Return the sum of values in the given double array, ignoring NaN
	 * @param values
	 * @return
	 */
	public static AccumulatorResult sum(double[] values){
		double result = 0;
		for(double value: values){
			if(Double.isNaN(value)) continue;
			result += value;
		}
		return new AccumulatorResult(result);
	}

	/**
	 * Return the sum of values in the given double array, in log space, ignoring NaN
	 * @param values
	 * @return
	 */
	public static AccumulatorResult sumInLogSpace(double[] values){
		double result = 0;
		double max = max(values).value;
		for(double value: values){
			if(Double.isNaN(value)) continue;
			if(value != Double.NEGATIVE_INFINITY){
				result += Math.exp(value-max);
			}
		}
		return new AccumulatorResult(Math.log(result)+max);
	}

	/**
	 * Return the maximum value in the given double array, ignoring NaN.
	 * Will also set the index of the maximum value in the array
	 * @param values
	 * @return
	 */
	public static AccumulatorResult max(double[] values){
		double result = Double.NEGATIVE_INFINITY;
		int parentIdx = -1;
		for(int i=0; i<values.length; i++){
			if(Double.isNaN(values[i])) continue;
			if(Double.compare(values[i], result) > 0){
				result = values[i];
				parentIdx = i;
			}
		}
		return new AccumulatorResult(result, parentIdx);
	}

	/**
	 * Return a double array with negated values
	 * @param values
	 * @return
	 */
	public static double[] negate(double[] values){
		double[] result = new double[values.length];
		for(int i=0; i<values.length; i++){
			result[i] = -values[i];
		}
		return result;
	}

}
