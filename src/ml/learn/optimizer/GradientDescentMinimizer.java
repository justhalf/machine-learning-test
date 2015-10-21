package ml.learn.optimizer;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import lbfgsb.FunctionValues;
import ml.learn.linear.CRF.LogLikelihood;
import ml.learn.linear.Instance;

public class GradientDescentMinimizer {
		
	public enum LearningRate {
		OPTIMAL, CONSTANT, INVT
	};
	
	private LearningRate learningRate;
	private double eta0; // initial learning rate
	private double alpha; // regularization term
	private double gamma = 0.1; // eta0 multiplier
	private int timestep = 10000; // update eta every n timestep
	private int iterations;
	private int batchSize;
	private static Random random = new Random(1);
	
	public GradientDescentMinimizer() {
		// defaults to SGD
		this(LearningRate.CONSTANT, 0.01, 1.0, 50, 1);
	}

	public GradientDescentMinimizer(LearningRate learningRate, double eta0, double alpha, int iterations, int batchSize) {
		setLearningRate(learningRate);
		setEta0(eta0);
		setAlpha(alpha);
		setIterations(iterations);
		setBatchSize(batchSize);
	}

	public GradientDescentMinimizer(double eta0, int iterations) {
		this(LearningRate.CONSTANT, eta0, 1.0, iterations, 1);
	}
	
	public double[] minimize(LogLikelihood fun, double[] startingWeights) {
		boolean useNumeric = false;
		int dim = startingWeights.length;
		double weights[] = startingWeights.clone();
		List<Instance> trainingData = new ArrayList<Instance>(fun.staticTrainingData);
		System.out.println("Training data size = "+trainingData.size());
		System.out.println("Learning rate = "+learningRate);
		System.out.println("eta0 = "+eta0);
		System.out.println("Iterations = "+iterations);
		System.out.println("Batch size = "+batchSize);
		long startTime = System.currentTimeMillis();
		int cnt = 0;
		for (int i = 0; i < iterations; i++) {
			Collections.shuffle(trainingData);
			for (int j = 0; j < trainingData.size(); j+=batchSize) {
				int endIdx = Math.min(j+batchSize, trainingData.size());
				List<Instance> subData = trainingData.subList(j, endIdx);
				FunctionValues valGrad = fun.getValues(weights, subData);
				double[] numGrads = new double[dim];
				if (useNumeric) numGrads = numericalGradient(fun, weights, subData);
				long endTime = System.currentTimeMillis();
				if (iterations*trainingData.size() <= 500 || cnt % 500 == 0) {
					FunctionValues curF = fun.getValues(weights, fun.staticTrainingData);
					System.out.printf("Iteration %d.%d: %.3f, %.3fs elapsed\n", (i+1), (j+1), (-curF.functionValue), (endTime-startTime)/1000.0);
				}
				for (int k = 0; k < dim; k++) {
					if (!useNumeric) {
						weights[k] = weights[k] - getEtaT(i*trainingData.size()+j)/batchSize*valGrad.gradient[k];
					} else {
						weights[k] = weights[k] - getEtaT(i*trainingData.size()+j)/batchSize*numGrads[k];
					}
				}
				cnt++;
			}
		}
		return weights;
	}
	
	public double[] minimize2(LogLikelihood fun, double[] startingWeights) {
		boolean useNumeric = false;
		int dim = startingWeights.length;
		double weights[] = startingWeights.clone();
		List<Instance> trainingData = fun.staticTrainingData;

		System.out.println("Training data size = "+trainingData.size());
		System.out.println("Learning rate = "+learningRate);
		if (learningRate != LearningRate.INVT)
			System.out.println("eta0 = "+eta0);
		System.out.println("Iterations = "+iterations);
		System.out.println("Batch size = "+batchSize);
		
		long startTime = System.currentTimeMillis();
		for (int i = 0; i < iterations; i++) {
			List<Instance> sampleData = new ArrayList<Instance>();
			if (batchSize == trainingData.size()) {
				sampleData = trainingData;
			} else {
				for (int j = 0; j < batchSize; j++) {
					int idx = random.nextInt(trainingData.size());
					sampleData.add(trainingData.get(idx));
				}
			}
			FunctionValues valGrad = fun.getValues(weights, sampleData);
			double[] numGrads = new double[dim];
			if (useNumeric) numGrads = numericalGradient(fun, weights, sampleData);
			long endTime = System.currentTimeMillis();
			if (iterations <= 500 || i % 500 == 0) {
				FunctionValues curF = fun.getValues(weights, fun.staticTrainingData);
				System.out.printf("Iteration %d: %.3f, %.3fs elapsed\n", (i+1), (-curF.functionValue), (endTime-startTime)/1000.0);
			}
			
			for (int j = 0; j < dim; j++) {
				if (!useNumeric) {
					weights[j] = weights[j] - getEtaT(i)/batchSize*valGrad.gradient[j];
				} else {
					weights[j] = weights[j] - getEtaT(i)/batchSize*numGrads[j];
				}
			}
		}
		return weights;
	}
	
	private double getEtaT(int t) {
		if (learningRate == LearningRate.OPTIMAL) {
			return eta0*Math.pow(gamma, t/timestep);
			//return eta0/(1+alpha*eta0*t);
		} else if (learningRate == LearningRate.INVT) {
			return 1.0/(t+1);
		}
		return eta0; // constant eta
	}
	
	private double[] numericalGradient(LogLikelihood fun, double[] startingWeights, List<Instance> trainingData) {
		int dim = startingWeights.length;
		double[] grads = new double[dim];
		double EPS = 0.000000001;
		for (int i = 0; i < dim; i++) {
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
	
	private void gradientCheck(LogLikelihood fun, double[] startingWeights, List<Instance> trainingData) {
		int dim = startingWeights.length;
		double EPS = 0.000000001;
		FunctionValues F = fun.getValues(startingWeights, trainingData);
		double numGrads[] = numericalGradient(fun, startingWeights, trainingData);
		for (int i = 0; i < dim; i++) {
			System.out.print("Weight #"+i+" "+fun.getReverseFeatureIndices()[i]+" ... ");
			
			double[] minEpWeights = startingWeights.clone();
			minEpWeights[i] -= EPS;
			double[] plusEpWeights = startingWeights.clone();
			plusEpWeights[i] += EPS;
			FunctionValues minF = fun.getValues(minEpWeights, trainingData);
			FunctionValues plusF = fun.getValues(plusEpWeights, trainingData);
			double grad = (plusF.functionValue-minF.functionValue)/(2*EPS);
			
			if (relativeError(F.gradient[i], grad) > 0.01) {
				System.out.print("Checking weight #"+i+" "+fun.getReverseFeatureIndices()[i]+" ... ");
				System.out.println("WRONG");
				System.out.printf("Explicit gradient values: %f\n", F.gradient[i]);
				System.out.printf("Numerical gradient value: %f\n", numGrads[i]);
			} else {
				System.out.print("Checking weight #"+i+" "+fun.getReverseFeatureIndices()[i]+" ... ");
				System.out.println("OK");
			}
		}
	}
	
	private double relativeError(double f1, double f2) {
		double abs1 = Math.abs(f1);
		double abs2 = Math.abs(f2);
		double diff = Math.abs(f1-f2);
		return diff/Math.max(abs1, abs2);
	}
	
	public LearningRate getLearningRate() {
		return learningRate;
	}

	public void setLearningRate(LearningRate learningRate) {
		this.learningRate = learningRate;
	}

	public double getEta0() {
		return eta0;
	}

	public void setEta0(double eta0) {
		this.eta0 = eta0;
	}

	public double getAlpha() {
		return alpha;
	}

	public void setAlpha(double alpha) {
		this.alpha = alpha;
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
