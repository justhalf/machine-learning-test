package ml.learn.optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import lbfgsb.FunctionValues;
import ml.learn.linear.CRF.LogLikelihood;
import ml.learn.linear.Instance;

public class GradientDescentMinimizer {
		
	public enum LearningAdjustment {
		OPTIMAL, CONSTANT, INVT, INTERVAL
	};
	
	private LearningAdjustment learningAdjustment;
	public int interval = 1; // for printing
	public int maxIterPrintAll = 100; 
	private double eta0; // initial learning rate
	private double alpha; // regularization term
	private double gamma; // eta0 multiplier
	private int timestep; // update eta every n timestep		
	private double momentum;
	private boolean averaged;
	private boolean adaGrad;

	private int iterations;
	private int batchSize;
	private static Random random = new Random(1);
	private static final boolean CHECK_GRADIENT = false;
	
	public GradientDescentMinimizer() {
		// defaults to SGD
		this(LearningAdjustment.CONSTANT, 0.01, 1.0, 50, 1);
	}

	public GradientDescentMinimizer(LearningAdjustment learningAdjustment, double eta0, double alpha, int iterations, int batchSize) {
		setLearningAdjustment(learningAdjustment);
		setEta0(eta0);
		setAlpha(alpha);
		setIterations(iterations);
		setBatchSize(batchSize);
	}

	public GradientDescentMinimizer(double eta0, int iterations) {
		this(LearningAdjustment.CONSTANT, eta0, 1.0, iterations, 1);
	}
	
	public double[] minimize(LogLikelihood fun, double[] startingWeights) {
		if(CHECK_GRADIENT){
			gradientCheck(fun, startingWeights, fun.staticTrainingData);
		}
		boolean useNumeric = false;
		int dim = startingWeights.length;
		double weights[] = startingWeights.clone();
		List<Instance> trainingData = fun.staticTrainingData;

		System.out.println("Training data size = "+trainingData.size());
		System.out.println("Learning rate = "+learningAdjustment);
		if (learningAdjustment != LearningAdjustment.INVT)
			System.out.println("eta0 = "+eta0);
		if (learningAdjustment == LearningAdjustment.INTERVAL) {
			System.out.println("Timestep = "+timestep);
			System.out.println("Gamma = "+gamma);
		}
		System.out.println("Iterations = "+iterations);
		System.out.println("Batch size = "+batchSize);
		
		long startTime = System.currentTimeMillis();
		double prevVal = 0.0;
		double diff = 0.0;
		String suffix = "";
		
		double prevUpdate[] = new double[weights.length];
		double sumWeights[] = startingWeights.clone();
		double sqrSumWeights[] = startingWeights.clone();
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
			if (iterations <= maxIterPrintAll || i % interval == 0) {
				double curWeights[] = weights.clone();
				if (averaged) {
					for (int j = 0; j < dim; j++) {
						curWeights[j] = sumWeights[j]/(i+1);
					}
				}
				double curVal = -fun.getValues(curWeights).functionValue;
				if(prevVal == 0.0){
					diff = curVal - prevVal;
					suffix = "";
				} else {
					diff = 100.0*(curVal - prevVal)/Math.abs(prevVal);
					suffix = "%";
				}
				System.out.printf("Iteration %d: %.3f (%+.6f%s), %.3fs elapsed\n", i+1, curVal, diff, suffix, (endTime-startTime)/1000.0);
//				if(diff < 0){
//					eta0 /= 1.25;
//				}
				prevVal = curVal;
			}
			for (int j = 0; j < dim; j++) {
				double gradJ = useNumeric ? numGrads[j] : valGrad.gradient[j];
				double newUpdateJ;
				if (adaGrad) {
					newUpdateJ = - getEtaT(i)/batchSize*gradJ/(Math.sqrt(sqrSumWeights[j]));
				} else {
					newUpdateJ = momentum*prevUpdate[j] - getEtaT(i)/batchSize*gradJ;
				}
				weights[j] += newUpdateJ;
//				if (!useNumeric) {
//					weights[j] = weights[j] - getEtaT(i)/batchSize*valGrad.gradient[j];
//				} else {
//					weights[j] = weights[j] - getEtaT(i)/batchSize*numGrads[j];
//				}
				prevUpdate[j] = newUpdateJ;
				sumWeights[j] += weights[j];
				sqrSumWeights[j] += (gradJ*gradJ);
			}
		}
		if (averaged) {
			for (int i = 0; i < dim; i++) {
				weights[i] = sumWeights[i]/iterations;
			}
		}
		return weights;
	}
	
	private double getEtaT(int t) {
		if (learningAdjustment == LearningAdjustment.OPTIMAL) {
			return eta0/(1+alpha*eta0*t);
		} else if (learningAdjustment == LearningAdjustment.INVT) {
			return 1.0/(t+1);
		} else if (learningAdjustment == LearningAdjustment.INTERVAL) {
			return eta0*Math.pow(gamma, t/timestep);
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
	
	public LearningAdjustment getLearningRate() {
		return learningAdjustment;
	}

	public void setLearningAdjustment(LearningAdjustment learningRate) {
		this.learningAdjustment = learningRate;
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
	
	public double getGamma() {
		return gamma;
	}

	public void setGamma(double gamma) {
		this.gamma = gamma;
	}

	public int getTimestep() {
		return timestep;
	}

	public void setTimestep(int timestep) {
		this.timestep = timestep;
	}
	
	public double getMomentum() {
		return momentum;
	}

	public void setMomentum(double momentum) {
		this.momentum = momentum;
	}
}
