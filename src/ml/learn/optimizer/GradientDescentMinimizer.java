package ml.learn.optimizer;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import lbfgsb.FunctionValues;
import ml.learn.linear.CRF.LogLikelihood;
import ml.learn.linear.CRF;
import ml.learn.linear.Instance;

public class GradientDescentMinimizer {
	
	public enum LearnType {
		STOCHASTIC, BATCH, MINIBATCH
	};
	
	private LearnType type = LearnType.STOCHASTIC; 
	private double learningRate = 0.01;
	private int iterations = 20;
	private int batchSize = 10;
	private static Random random = new Random(1);
	
	public GradientDescentMinimizer() {
		
	}
	
	public GradientDescentMinimizer(LearnType type, double learningRate, int iterations) {
		this.type = type;
		this.learningRate = learningRate;
		this.iterations = iterations;
	}
	
	public double[] minimize(LogLikelihood fun, double[] startingWeights) {
		//gradientCheck(fun, startingWeights);
		boolean useNumeric = false;
		int dim = startingWeights.length;
		double weights[] = startingWeights.clone();
		List<Instance> trainingData = fun.staticTrainingData;
		if (type == LearnType.BATCH) {
			batchSize = trainingData.size(); 
		} else if (type == LearnType.STOCHASTIC) {
			batchSize = 1;
		}
		System.out.println("Training data size = "+trainingData.size());
		System.out.println("Learning type = "+type);
		System.out.println("Learning rate = "+learningRate);
		System.out.println("Iterations = "+iterations);
		System.out.println("Batch size = "+batchSize);
		long startTime = System.currentTimeMillis();
		for (int i = 0; i < iterations; i++) {
			for (int j = 0; j < trainingData.size(); j+=batchSize) {
				int endIdx = Math.min(j+batchSize, trainingData.size());
				List<Instance> subData = trainingData.subList(j, endIdx);
				FunctionValues valGrad = fun.getValues(weights, subData);
				double[] numGrads = new double[dim];
				if (useNumeric) numGrads = numericalGradient(fun, weights);
				long endTime = System.currentTimeMillis();
				System.out.printf("Iteration %d: %.3f, %.3fs elapsed\n", (i+1), (-valGrad.functionValue), (endTime-startTime)/1000.0);
				for (int k = 0; k < dim; k++) {
					if (!useNumeric) {
						weights[k] = weights[k] - learningRate/batchSize*valGrad.gradient[k];
					} else {
						weights[k] = weights[k] - learningRate/batchSize*numGrads[k];
					}
				}
//				learningRate = learningRate/(1+learningRate*i);
			}
		}
		return weights;
		//return startingWeights;
	}
	
	private double[] numericalGradient(LogLikelihood fun, double[] startingWeights) {
		int dim = startingWeights.length;
		double[] grads = new double[dim];
		double EPS = 0.000000001;
		for (int i = 0; i < dim; i++) {
			List<Instance> trainingData = fun.staticTrainingData;
			double[] minEpWeights = startingWeights.clone();
			minEpWeights[i] -= EPS;
			double[] plusEpWeights = startingWeights.clone();
			plusEpWeights[i] += EPS;
			FunctionValues minF = fun.getValues(minEpWeights, trainingData);
			FunctionValues plusF = fun.getValues(plusEpWeights, trainingData);
			double grad = (plusF.functionValue-minF.functionValue)/(2*EPS);
			grads[i] = grad;
		}
		return grads;
	}
	
	private void gradientCheck(LogLikelihood fun, double[] startingWeights) {
		int dim = startingWeights.length;
		double EPS = 0.000000001;
		List<Instance> trainingData = fun.staticTrainingData;
		for (int i = 0; i < dim; i++) {
			if (i != 36 && i != 0) continue;
			//System.out.print("Checking weight #"+i+" "+CRF.reverseFeatureIndices.get(i)+" ... ");
			double[] minEpWeights = startingWeights.clone();
			minEpWeights[i] -= EPS;
			double[] plusEpWeights = startingWeights.clone();
			plusEpWeights[i] += EPS;
			FunctionValues F = fun.getValues(startingWeights, trainingData);
			FunctionValues minF = fun.getValues(minEpWeights, trainingData);
			FunctionValues plusF = fun.getValues(plusEpWeights, trainingData);
			double grad = (plusF.functionValue-minF.functionValue)/(2*EPS);
			if (relativeError(F.gradient[i], grad) > 0.01) {
				System.out.print("Checking weight #"+i+" "+CRF.reverseFeatureIndices.get(i)+" ... ");
				System.out.println("WRONG");
				System.out.printf("y-eps,y,y+eps = %.14f, %.14f, %.14f\n", minF.functionValue, F.functionValue, plusF.functionValue);
				System.out.printf("Explicit gradient values: %.3f\n", F.gradient[i]);
				System.out.printf("Numerical gradient value: %.3f\n", grad);
			} else {
				System.out.print("Checking weight #"+i+" "+CRF.reverseFeatureIndices.get(i)+" ... ");
				System.out.println("OK");
				System.out.printf("y-eps,y,y+eps = %.14f, %.14f, %.14f\n", minF.functionValue, F.functionValue, plusF.functionValue);
				System.out.printf("Explicit gradient values: %.3f\n", F.gradient[i]);
				System.out.printf("Numerical gradient value: %.3f\n", grad);			
			}
		}
	}
	
	private double relativeError(double f1, double f2) {
		double abs1 = Math.abs(f1);
		double abs2 = Math.abs(f2);
		double diff = Math.abs(f1-f2);
		return diff/Math.max(abs1, abs2);
	}
	
	/*
	public double[] minimize(LogLikelihood fun, double[] startingWeights) {
		int dim = startingWeights.length;
		double weights[] = startingWeights.clone();
		for (int i = 0; i < iterations; i++) {
			List<Instance> trainingData;
			if (type == LearnType.STOCHASTIC) {
				batchSize = 1;
			}
			if (type == LearnType.BATCH) {
				trainingData = fun.staticTrainingData;
			} else {
				trainingData = sampleData(new ArrayList<Instance>(fun.staticTrainingData), batchSize);
			}
			double gradient[] = fun.getValues(weights, trainingData).gradient;
			for (int j = 0; j < dim; j++) {
				weights[j] = weights[j] - learningRate*gradient[j];
			}
		}
		return weights;
	}
	
	private List<Instance> sampleData(List<Instance> data, int size) {
		List<Instance> sample = new ArrayList<Instance>();
		for (int i = 0; i < size; i++) {
			int r = random.nextInt(data.size());
			sample.add(data.get(r));
		}
		return sample;
	}
	*/

	public LearnType getType() {
		return type;
	}

	public void setType(LearnType type) {
		this.type = type;
	}

	public double getLearningRate() {
		return learningRate;
	}

	public void setLearningRate(double learningRate) {
		this.learningRate = learningRate;
	}

	public int getIterations() {
		return iterations;
	}

	public void setIterations(int iterations) {
		this.iterations = iterations;
	}

	public int getBatchSize() {
		return batchSize;
	}

	public void setBatchSize(int batchSize) {
		this.batchSize = batchSize;
	}
}
