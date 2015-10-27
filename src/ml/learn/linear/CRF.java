package ml.learn.linear;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.function.Function;

import lbfgsb.DifferentiableFunction;
import lbfgsb.FunctionValues;
import lbfgsb.IterationFinishedListener;
import lbfgsb.LBFGSBException;
import lbfgsb.Minimizer;
import lbfgsb.Result;
import ml.learn.linear.Template.Feature;
import ml.learn.linear.Template.TagIndex;
import ml.learn.object.Tag;
import ml.learn.object.TaggedWord;
import ml.learn.optimizer.GradientDescentMinimizer;
import ml.learn.util.Common;
import ml.learn.optimizer.GradientDescentMinimizer.LearningAdjustment;

public class CRF implements StructuredClassifier{
	
	/** Indicating start of the sequence */
	public static final Tag START = Tag.START;
	/** Indicating end of the sequence */
	public static final Tag END = Tag.END;
	
	/** Indicate an unknown word */
	public static final String UNKNOWN_WORD = "-UNK-";
	/** Indicate a numeric */
	public static final String NUMERIC = "-NUM-";
	public static final String[] INIT_CAP_FEATURES = {"##NOINIT_NOCAP##", "##NOINIT_CAP##", "##INIT_NOCAP##", "##INIT_CAP##"};
	public static final String[] END_FEATURES = {"##END_S##", "##END_ED##", "##END_ING##", "##END_OTHER##"};
	
	/** The weights of current CRF model */
	public double[] weights;
	/** The regularization parameter sigma */
	public double regularizationParameter;
	
	/** Map of tags to indices */
	public LinkedHashMap<Tag, Integer> tags;
	/** An array of tags (mapping from indices to Tag objects) */
	public Tag[] reverseTags;
	
	/** Map of words to indices */
	public LinkedHashMap<String, Integer> words;
	/** An array of words (mapping from indices to words) */
	public String[] reverseWords;
	
	/**
	 * Mapping from full feature names (e.g., B-NP00:Confidence, START|B-NP) to index.
	 * The feature index corresponds with the weights in the respective index
	 */
	public Map<String,Integer> featureIndices;
	/** The reverse mapping of {@link #featureIndices} */
	public String[] reverseFeatureIndices;

	/** Mapping from partial feature names (e.g., U00:Confidence, B) to {@link TagIndex} */
	public Map<String, TagIndex> tagIndices;
	/** List of feature templates */
	public Template[] templates;
	
	/** The list of tag indices without START tag */
	public int[] noStart;
	/** The list of tag indices without the END tag */
	public int[] noEnd;
	/** The list of tag indices only for START tag */
	public int[] onlyStart;
	/** The list of tag indices only for END tag */
	public int[] onlyEnd;
	/** An empty array */
	public int[] empty;
	
	/** Whether to calculate forward backward in log space */
	public boolean useLogSpace = true;
	
	
	public Random random;
	
	public boolean useMultiThread = true;
	public int numThreads = 8;
	private ExecutorService executor = Executors.newFixedThreadPool(numThreads);
	
	// PARAMS
	boolean useSGD = true;
//	boolean useSGD = false;
	private double eta0 = 0.1;
	private int iterations = 100000;
	private int batchSize = 1;
//	private LearningAdjustment learningRate = LearningAdjustment.CONSTANT;
	private LearningAdjustment learningRate = LearningAdjustment.INVT;
	private double lambdaReg;
	private int timestep = 45000;
	private double gamma = 0.25;
	
	int interval = 1000; // for printing
	int maxIterPrintAll = 0;
	
	/**
	 * Create a CRF model with default feature templates
	 */
	public CRF(){
		this(new String[]{
//				"U00:%x[-2,0]",
//				"U01:%x[-1,0]",
				"U02:%x[0,0]",
//				"U03:%x[1,0]",
//				"U04:%x[2,0]",
//				"U05:%x[-1,0]/%x[0,0]",
//				"U06:%x[0,0]/%x[1,0]",
//
//				"U10:%x[-2,1]",
//				"U11:%x[-1,1]",
//				"U12:%x[0,1]",
//				"U13:%x[1,1]",
//				"U14:%x[2,1]",
//				"U15:%x[-2,1]/%x[-1,1]",
//				"U16:%x[-1,1]/%x[0,1]",
//				"U17:%x[0,1]/%x[1,1]",
//				"U18:%x[1,1]/%x[2,1]",
//
//				"U20:%x[-2,1]/%x[-1,1]/%x[0,1]",
//				"U21:%x[-1,1]/%x[0,1]/%x[1,1]",
//				"U22:%x[0,1]/%x[1,1]/%x[2,1]",
				
				"B",
				});
	}
	
	// 1,2,11,12 for NP chunking
	
	/**
	 * Create a CRF model with the feature templates taken from the specified file name.
	 * The file must contain one feature template per line. A feature template can start with either
	 * "U" or "B" representing unigram and bigram feature, and can contain zero or more macros in the form of 
	 * "%x[N,M]", where N is the position in the instance relative to current position, and M is the feature
	 * index (0 is usually for surface word)
	 * @param templateFile
	 * @throws IOException
	 */
	public CRF(String templateFile) throws IOException{
		InputStreamReader isr = new InputStreamReader(new FileInputStream(templateFile), "UTF-8");
		BufferedReader br = new BufferedReader(isr);
		List<Template> templates = new ArrayList<Template>();
		while(br.ready()){
			String line = br.readLine().trim();
			if(line.length() == 0 || line.startsWith("#")) continue;
			templates.add(new Template(line));
		}
		br.close();
		initialize(templates.toArray(new Template[templates.size()]));
	}
	
	/**
	 * Create a CRF model with the specified feature templates
	 * @param templates
	 */
	public CRF(String[] templates){
		Template[] templateObjects = new Template[templates.length];
		for(int i=0; i<templates.length; i++){
			templateObjects[i] = new Template(templates[i]);
		}
		initialize(templateObjects);
	}
	
	private void initialize(Template[] templates){
		random = new Random(0);
		tags = new LinkedHashMap<Tag, Integer>();
		words = new LinkedHashMap<String, Integer>();
		regularizationParameter = 1.0;
		this.templates = templates;
		this.tagIndices = new LinkedHashMap<String, TagIndex>();
	}
	
	private Result minimize(DifferentiableFunction function, double[] startingPoint) throws LBFGSBException{
		Minimizer alg = new Minimizer();
		alg.setNoBounds(startingPoint.length);
		alg.setIterationFinishedListener(new IterationFinishedListener(){
			
			public int i=0;
			public double lastValue = 0.0;
			public long startTime = System.currentTimeMillis();

			@Override
			public boolean iterationFinished(double[] point, double functionValue, double[] gradient) {
				i++;
				double change = -functionValue - lastValue;
				String suffix = "";
				if(lastValue != 0){
					change = 100.0*change/-lastValue;
					suffix = "%";
				}
				System.out.printf("Iteration %d: %.9f (%+.6f%s), elapsed time %.3fs\n", i, -functionValue, change, suffix, (System.currentTimeMillis()-startTime)/1000.0);
				lastValue = -functionValue;
//				gradient = negate(gradient);
//				System.out.println("Point:");
//				for(int j=0; j<point.length; j++){
//					System.out.printf("%.4f ", point[j]);
//				}
//				System.out.println();
//				System.out.println("Gradient:");
//				for(int j=0; j<point.length; j++){
//					System.out.printf("%.4f ", point[j]);
//				}
//				System.out.println();
				return true;
			}
			
		});
		return alg.run(function, startingPoint);
	}
	
	/**
	 * A loglikelihood function
	 * @author Aldrian Obaja <aldrianobaja.m@gmail.com>
	 *
	 */
	public class LogLikelihood implements DifferentiableFunction{
		
		public List<Instance> trainingData;
		public List<Instance> staticTrainingData;
//		public LinkedHashMap<Instance, double[][]> forwards;
//		public LinkedHashMap<Instance, double[][]> backwards;
		public Map<Instance, double[][]> forwards;
		public Map<Instance, double[][]> backwards;
		public double[] empiricalDistribution;
		public double[] staticEmpiricalDistribution;
		
		public String[] getReverseFeatureIndices(){
			return reverseFeatureIndices;
		}
		
		public LogLikelihood(List<Instance> trainingData){
			this.staticTrainingData = trainingData;
			this.trainingData = trainingData;
			empiricalDistribution = computeEmpiricalDistribution();
			staticEmpiricalDistribution = empiricalDistribution;
		}

		private double[] computeEmpiricalDistribution(){
			double[] result = new double[featureIndices.size()];
			Arrays.fill(result, 0);
			int size = trainingData.size();
			int total = 0;
			for(Instance instance: trainingData){
				int n = instance.words.size()+2;
				for(int j=0; j<n-1; j++){
					Tag prevTag = instance.getTagAt(j-1);
					Tag curTag = instance.getTagAt(j);
					for(int i: featuresActivated(instance, j, prevTag, curTag)){
						if(i == -1) continue;
						result[i] += 1;
					}
				}
				total++;
				if(total % 10000 == 0){
					System.out.println(String.format("Completed %d/%d", total, size));
				}
			}
			return result;
		}
		
		@Override
		public FunctionValues getValues(double[] point){
			return getValues(point, staticTrainingData);
		}

		public FunctionValues getValues(double[] point, List<Instance> trainingData) {
			this.trainingData = trainingData;
			if(trainingData != staticTrainingData){
				empiricalDistribution = computeEmpiricalDistribution();
			} else {
				empiricalDistribution = staticEmpiricalDistribution;
			}
//			System.out.println("Computing forward backward...");
//			long startTime = System.currentTimeMillis();
			computeForwardBackward(point);
//			long endTime = System.currentTimeMillis();
//			System.out.printf("Done forward-backward in %.3fs\n", (endTime-startTime)/1000.0);
			double value = 0.0;
//			startTime = System.currentTimeMillis();
			List<Callable<Object>> tasks = new ArrayList<Callable<Object>>();
			for(Instance instance: trainingData){
				if (useMultiThread) {
					tasks.add(new Callable<Object>() {

						@Override
						public Object call() throws Exception {
							double instVal = 0.0;
							int n = instance.words.size()+2;
							for(int position=0; position<n-1; position++){
								Tag prevTag = instance.getTagAt(position-1);
								Tag curTag = instance.getTagAt(position);
								for(int i: featuresActivated(instance, position, prevTag, curTag)){
									if(i == -1) continue;
									instVal += point[i];
								}
							}
							if(useLogSpace){
								instVal -= normalizationConstant(instance);
							} else {
								instVal -= Math.log(normalizationConstant(instance));
							}
							return instVal;
						}
					});
				} else {
					int n = instance.words.size()+2;
					for(int position=0; position<n-1; position++){
						Tag prevTag = instance.getTagAt(position-1);
						Tag curTag = instance.getTagAt(position);
						for(int i: featuresActivated(instance, position, prevTag, curTag)){
							if(i == -1) continue;
							value += point[i];
						}
					}
					if(useLogSpace){
						value -= normalizationConstant(instance);
					} else {
						value -= Math.log(normalizationConstant(instance));
					}
				}
			}
			if (useMultiThread) {
				try {
					List<Future<Object>> result = executor.invokeAll(tasks);
					for (Future<Object> f: result) {
						value += (double) f.get();
					}
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
			value -= regularizationTerm(point);
//			endTime = System.currentTimeMillis();
//			System.out.printf("Done calculating value in %.3fs\n", (endTime-startTime)/1000.0);
//			startTime = System.currentTimeMillis();
			double[] gradient = computeGradient(point);
//			endTime = System.currentTimeMillis();
//			System.out.printf("Done calculating gradient in %.3fs\n", (endTime-startTime)/1000.0);
			
			// Values and gradient are negated to find the maximum
			return new FunctionValues(-value, Common.negate(gradient));
		}
		
		/**
		 * Compute lambda^2/(2.sigma^2) for calculating likelihood value
		 * @return
		 */
		private double regularizationTerm(double[] point){
			double result = 0;
			double scale = ((double)trainingData.size())/staticTrainingData.size();
			for(int i=0; i<point.length; i++){
				result += Math.pow(point[i], 2);
			}
			result /= 2*Math.pow(regularizationParameter, 2)*scale;
			return result;
		}
				
		private void computeForwardBackward(double[] point){
			if (useMultiThread) {
				forwards = Collections.synchronizedMap(new LinkedHashMap<Instance, double[][]>());
				backwards = Collections.synchronizedMap(new LinkedHashMap<Instance, double[][]>());
			} else {
				forwards = new LinkedHashMap<Instance, double[][]>();
				backwards = new LinkedHashMap<Instance, double[][]>();
			}
//			int size = trainingData.size();
//			int total = 0;
			
			List<Callable<Object>> tasks = new ArrayList<Callable<Object>>();
			for(Instance instance: trainingData){
				if (useMultiThread) {
					tasks.add(new Callable<Object>() {
	
						@Override
						public Object call() throws Exception {
							int n = instance.words.size()+2;
							double[][] forward = new double[n][tags.size()];
							double[][] backward = new double[n][tags.size()];
							Function<double[], Common.AccumulatorResult> summaryFunction;
							if(useLogSpace){
								summaryFunction = Common::sumInLogSpace;
								Arrays.fill(forward[0], Double.NEGATIVE_INFINITY);
								Arrays.fill(backward[n-1], Double.NEGATIVE_INFINITY);
								forward[0][tags.get(START)] = 0;
								backward[n-1][tags.get(END)] = 0;
							} else {
								summaryFunction = Common::sum;
								Arrays.fill(forward[0], 0);
								Arrays.fill(backward[n-1], 0);
								forward[0][tags.get(START)] = 1;
								backward[n-1][tags.get(END)] = 1;
							}
							fillValues(instance, forward, true, point, summaryFunction, null);
							fillValues(instance, backward, false, point, summaryFunction, null);
							forwards.put(instance, forward);
							backwards.put(instance, backward);
							return null;
						}
						
					});
				} else {
					int n = instance.words.size()+2;
					double[][] forward = new double[n][tags.size()];
					double[][] backward = new double[n][tags.size()];
					Function<double[], Common.AccumulatorResult> summaryFunction;
					if(useLogSpace){
						summaryFunction = Common::sumInLogSpace;
						Arrays.fill(forward[0], Double.NEGATIVE_INFINITY);
						Arrays.fill(backward[n-1], Double.NEGATIVE_INFINITY);
						forward[0][tags.get(START)] = 0;
						backward[n-1][tags.get(END)] = 0;
					} else {
						summaryFunction = Common::sum;
						Arrays.fill(forward[0], 0);
						Arrays.fill(backward[n-1], 0);
						forward[0][tags.get(START)] = 1;
						backward[n-1][tags.get(END)] = 1;
					}
					fillValues(instance, forward, true, point, summaryFunction, null);
					fillValues(instance, backward, false, point, summaryFunction, null);
					forwards.put(instance, forward);
					backwards.put(instance, backward);
//					System.out.println(instance);
//					System.out.println("Forward:");
//					for(int j=0; j<tags.size(); j++){
//						System.out.printf("%15s ", reverseTags[j]);
//					}
//					System.out.println();
//					for(int i=0; i<n; i++){
//						for(int j=0; j<tags.size(); j++){
//							System.out.printf("%15.2f ", forward[i][j]);
//						}
//						System.out.println();
//					}
//					System.out.println("Backward:");
//					for(int i=0; i<n; i++){
//						for(int j=0; j<tags.size(); j++){
//							System.out.printf("%15.2f ", backward[i][j]);
//						}
//						System.out.println();
//					}
//					total++;
//					if(total % 100 == 0){
//						System.out.println(String.format("Completed %d/%d", total, size));
//					}
				}
			}
			if (useMultiThread) {
				try {
					executor.invokeAll(tasks);
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
			}
		}
		
		private double[] computeGradient(double[] point){
			double[] result = new double[point.length];
//			System.out.println("Computing model distribution...");
			double[] modelDistribution = computeModelDistribution(point);
//			System.out.println("Computing regularization...");
			double[] regularization = computeGradientOfRegularization(point);
			for(int i=0; i<result.length; i++){
				result[i] = empiricalDistribution[i] - modelDistribution[i] - regularization[i];
//				if (true) {
//					System.out.println("Statistics for "+i);
//					System.out.println("Empirical "+empiricalDistribution[i]);
//					System.out.println("Model "+modelDistribution[i]);
//					System.out.println("Reg "+regularization[i]);
//					if (i == 36)
//					System.out.println("Predicted model dist: "+ (empiricalDistribution[i]-regularization[i]-0.094));
//				}
//				System.out.printf("%.3f ", result[i]);
			}
//			System.out.println();
			return result;
		}
		
		/**
		 * Compute model distribution
		 * sum_trainingdata 1/normalization (sum j=1 to n sum_tags sum_nextTag f_i(s,s',x,j)*forward_j(s|x)*factor(x,s,s')*backward_j+1(s'|x))
		 * @param point
		 * @return
		 */
		private double[] computeModelDistribution(double[] point){
			double[] result = new double[featureIndices.size()];
			Arrays.fill(result, 0);
			List<Callable<Object>> tasks = new ArrayList<Callable<Object>>();
			for(Instance instance: trainingData){
				if (useMultiThread) {
					tasks.add(new Callable<Object>() {

						@Override
						public Object call() throws Exception {
							double[] instanceExpectation = new double[featureIndices.size()];
//							Arrays.fill(instanceExpectation, 0);
							int n = instance.words.size()+2;
							double[][] forward = forwards.get(instance);
							double[][] backward = backwards.get(instance);
							for(int j=0; j<n-1; j++){
								for(Tag curTag: tags.keySet()){
									int curTagIdx = tags.get(curTag);
									for(int nextTagIdx: getNextTags(curTag, j, n)){
										Tag nextTag = reverseTags[nextTagIdx];
										double factor;
										if(useLogSpace){
											factor = computeFactorInLogSpace(point, instance, j, curTag, nextTag);
										} else {
											factor = computeFactor(point, instance, j, curTag, nextTag);
										}
										for(int i: featuresActivated(instance, j, curTag, nextTag)){
											if(i == -1) continue;
											if(useLogSpace){
												instanceExpectation[i] += Math.exp(forward[j][curTagIdx]+factor+backward[j+1][nextTagIdx]-backward[0][tags.get(START)]);
											} else {
												instanceExpectation[i] += forward[j][curTagIdx]*factor*backward[j+1][nextTagIdx]/backward[0][tags.get(START)];
											}
										}
									}
								}
							}
							for(int i=0; i<featureIndices.size(); i++){
								synchronized (result) {
									result[i] += instanceExpectation[i];
								}
							}
							return null;
						}
					});
				} else {
					double[] instanceExpectation = new double[featureIndices.size()];
//					Arrays.fill(instanceExpectation, 0);
					int n = instance.words.size()+2;
					double[][] forward = forwards.get(instance);
					double[][] backward = backwards.get(instance);
					for(int j=0; j<n-1; j++){
						for(Tag curTag: tags.keySet()){
							int curTagIdx = tags.get(curTag);
							for(int nextTagIdx: getNextTags(curTag, j, n)){
								Tag nextTag = reverseTags[nextTagIdx];
								double factor;
								if(useLogSpace){
									factor = computeFactorInLogSpace(point, instance, j, curTag, nextTag);
								} else {
									factor = computeFactor(point, instance, j, curTag, nextTag);
								}
								for(int i: featuresActivated(instance, j, curTag, nextTag)){
									if(i == -1) continue;
									if(useLogSpace){
										instanceExpectation[i] += Math.exp(forward[j][curTagIdx]+factor+backward[j+1][nextTagIdx]-backward[0][tags.get(START)]);
									} else {
										instanceExpectation[i] += forward[j][curTagIdx]*factor*backward[j+1][nextTagIdx]/backward[0][tags.get(START)];
									}
								}
							}
						}
					}
					for(int i=0; i<featureIndices.size(); i++){
						result[i] += instanceExpectation[i];
					}
				}
			}
			if (useMultiThread) {
				try {
					executor.invokeAll(tasks);
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
			}
			return result;
		}
		
		private double computeFactor(double[] point, Instance instance, int position, Tag prevTag, Tag curTag){
			return Math.exp(computeFactorInLogSpace(point, instance, position, prevTag, curTag));
		}
		
		private double computeFactorInLogSpace(double[] point, Instance instance, int position, Tag prevTag, Tag curTag){
			double result = 0;
			for(int i: featuresActivated(instance, position, prevTag, curTag)){
				if(i == -1) continue;
				result += point[i];
			}
			return result;
		}
		
		/**
		 * Compute lambda/sigma^2 for calculating regularization in gradient
		 * @param point
		 * @return
		 */
		private double[] computeGradientOfRegularization(double[] point){
			double scale = ((double)trainingData.size())/staticTrainingData.size();
			double[] result = new double[featureIndices.size()];
			for(int i=0; i<result.length; i++){
				result[i] = point[i]/Math.pow(regularizationParameter, 2)*scale;
			}
			return result;
		}
		
		/**
		 * The value of Z for a specific instance
		 * @param instance
		 * @return
		 */
		private double normalizationConstant(Instance instance){
			return backwards.get(instance)[0][tags.get(START)];
		}
	}
	
	private int[] getNextTags(Tag curTag, int position, int n){
		if(curTag == END){
			return empty;
		}
		if(position == n-2){
			return onlyEnd;
		}
		if(position == 0){
			if(curTag == START){
				return noStart;
			}
			return empty;
		}
		return noStart;
	}
	
	private int[] getPreviousTags(Tag curTag, int position, int n){
		if(curTag == START){
			return empty;
		}
		if(position == 1){
			return onlyStart;
		}
		if(position == n-1){
			if(curTag == END){
				return noEnd;
			}
			return empty;
		}
		return noEnd;
	}
	
	private void fillValues(Instance instance, double[][] lattice, boolean isForward, double[] weights, Function<double[], Common.AccumulatorResult> accumulator, int[][] parentIdx){
		int n = instance.words.size()+2;
		int start, end, step;
		if(isForward){
			start = 1;
			end = n-1;
			step = 1;
		} else {
			start = n-2;
			end = 0;
			step = -1;
		}
		int tagSize = lattice[0].length;
		int instanceLength = lattice.length;
		double[] values = new double[tagSize];
		double value=0;
		for(int j=start; j*step<=end*step; j+=step){
			double[] prevValues = lattice[j-step];
			double[] curValues = lattice[j];
			int[] curParentIdx = (parentIdx != null) ? parentIdx[j] : null;
			for(int curTagIdx=0; curTagIdx<tagSize; curTagIdx++){
				Tag curTag = reverseTags[curTagIdx];
				Arrays.fill(values, Double.NaN);
				int[] reachableTags;
				if(isForward){
					reachableTags = getPreviousTags(curTag, j, instanceLength);
				} else {
					reachableTags = getNextTags(curTag, j, instanceLength);
				}
				for(int reachableTagIdx: reachableTags){
					Tag reachableTag = reverseTags[reachableTagIdx];
					Tag prevTagArg, curTagArg;
					int position;
					if(isForward){
						prevTagArg = reachableTag;
						curTagArg = curTag;
						position = j-1;
					} else {
						prevTagArg = curTag;
						curTagArg = reachableTag;
						position = j;
					}
					value = 0.0;
					for(int i: featuresActivated(instance, position, prevTagArg, curTagArg)){
						if(i == -1) continue;
						value += weights[i];
					}
//					if(!isForward){
//						System.out.printf("Pos: %d, Prev: %s, Cur: %s, Value: %.3f\n", position, prevTagArg, curTagArg, value);
//					}
					if(useLogSpace){
						values[reachableTagIdx] = prevValues[reachableTagIdx]+value;
					} else {
						values[reachableTagIdx] = prevValues[reachableTagIdx]*Math.exp(value);
					}
				}
//				Common.AccumulatorResult result;
//				if (isTraining) {
//					if (useLogSpace)
//						result = Common.sumInLogSpace(values);
//					else
//						result = Common.sum(values);
//				} else {
//					result = Common.max(values);
//				}
				Common.AccumulatorResult result = accumulator.apply(values);
				curValues[curTagIdx] = result.value;
				if(curParentIdx != null){
					curParentIdx[curTagIdx] = result.maxIdx;
				}
				
			}
		}
	}
	
	/**
	 * First pass of the training data to get the number of words and tags
	 * @param trainingData
	 */
	private void readTrainingData(List<Instance> trainingData){
		for(Instance instance: trainingData){
			for(TaggedWord wordTag: instance.words){
				String word = wordTag.word();
				Tag tag = wordTag.tag();
				word = normalize(word, true);
				if(!words.containsKey(word)){
					words.put(word, words.size());
				}
				if(!tags.containsKey(tag)){
					tags.put(tag, tags.size());
				}
			}
		}
		tags.put(START, tags.size());
		tags.put(END, tags.size());
		words.put(UNKNOWN_WORD, words.size());
		
//		for(String feature: INIT_CAP_FEATURES){
//			words.put(feature, words.size());
//		}
//		for(String feature: END_FEATURES){
//			words.put(feature, words.size());
//		}
		
		reverseTags = new Tag[tags.size()];
		for(Tag tag: tags.keySet()){
			reverseTags[tags.get(tag)] = tag;
		}
		reverseWords = new String[words.size()];
		for(String word: words.keySet()){
			reverseWords[words.get(word)] = word;
		}
		
		noStart = new int[tags.size()-1];
		noEnd = new int[tags.size()-1];
		onlyStart = new int[]{tags.get(START)};
		onlyEnd = new int[]{tags.get(END)};
		empty = new int[0];
		int idx = 0;
		for(Tag tag: tags.keySet()){
			if(tag != START){
				noStart[idx] = tags.get(tag);
				idx++;
			}
		}
		idx = 0;
		for(Tag tag: tags.keySet()){
			if(tag != END){
				noEnd[idx] = tags.get(tag);
				idx++;
			}
		}
	}

	private void buildFeatures(List<Instance> trainingData, boolean isTraining) {
		if(isTraining){
			featureIndices = new HashMap<String, Integer>();
		}
		for(Instance instance: trainingData){
			List<TaggedWord> wordTags = instance.words;
			for(int position=0; position<wordTags.size()+1; position++){
				Tag prevTag = instance.getTagAt(position-1);
				Tag curTag = instance.getTagAt(position);
				insertFeatures(instance, position, prevTag, curTag, isTraining);
			}
		}
		if(isTraining){
			reverseFeatureIndices = new String[featureIndices.size()];
			for(String feature: featureIndices.keySet()){
				reverseFeatureIndices[featureIndices.get(feature)] = feature;
			}
			System.out.println("Num of featureIndices: "+featureIndices.size());
			System.out.println("Num of tags: "+tags.size());
		}
	}
	
	private void insertFeatures(Instance instance, int position, Tag prevTag, Tag curTag, boolean isTraining){
		Feature feature = null;
		String featureName, featureWithoutTag;
		List<Feature> features = new ArrayList<Feature>();
		for(int templateIdx=0; templateIdx<templates.length; templateIdx++){
			feature = templates[templateIdx].getFeature(instance, position, prevTag, curTag);
			featureName = feature.featureName;
			featureWithoutTag = feature.featureWithoutTag;
			if(isTraining){
				if(!featureIndices.containsKey(featureName)){
					featureIndices.put(featureName, featureIndices.size());
				}
				if(!tagIndices.containsKey(featureWithoutTag)){
					tagIndices.put(featureWithoutTag, new TagIndex());
				}
				feature.tagIndex = tagIndices.get(featureWithoutTag);
				feature.addTag(prevTag, curTag, featureIndices.get(featureName));
				features.add(feature);
			} else if (featureIndices.containsKey(featureName)){
				feature.tagIndex = tagIndices.get(featureWithoutTag);
				features.add(feature);
			}
		}
		instance.features[position] = features.toArray(new Feature[features.size()]);
	}
	
	/**
	 * Returns the list of feature indices activated in the specified instance at the specified position
	 * with the specified previous tag and current tag.
	 * @param instance
	 * @param position
	 * @param prevTag
	 * @param curTag
	 * @return
	 */
	private int[] featuresActivated(Instance instance, int position, Tag prevTag, Tag curTag){
		int[] result = new int[instance.features[position].length];
		int idx = 0;
		for(Feature feature: instance.features[position]){
//			if(feature.present(prevTag, curTag)){
				result[idx] = feature.getFeatureIndex(prevTag, curTag);
//			} else {
//				result[idx] = -1;
//			}
			idx++;
		}
		return result;
	}
	
	/**
	 * Normalize the word to the canonical representation (e.g., all numbers to a single token)
	 * @param word
	 * @param isTraining
	 * @return
	 */
	private String normalize(String word, boolean isTraining){
		int digits = 0;
		for(char c: word.toCharArray()){
			if(c >= 48 && c <= 57){
				digits++;
			}
		}
		if(word.matches("[0-9]+([0-9,-.])*") || digits*3 >= word.length()*2){
			word = NUMERIC;
		}
		if(!isTraining && !words.containsKey(word)) return UNKNOWN_WORD;
		return word;
	}

	@Override
	public void train(List<Instance> trainingData) {
		System.out.println("Reading training data...");
		readTrainingData(trainingData);
		System.out.println("Building features...");
		buildFeatures(trainingData, true);
		System.out.println("Preparing for minimization...");
		LogLikelihood logLikelihood = new LogLikelihood(trainingData);
		double[] startingPoint = new double[featureIndices.size()];
		System.out.println("Starting point:");
		
		for(int i=0; i<startingPoint.length; i++){
			startingPoint[i] = random.nextGaussian();
			System.out.printf("%.3f ", startingPoint[i]);
		};
		System.out.println();
//		printGradientAt(logLikelihood, startingPoint);
//		System.out.println();
//		System.out.println("Starting point:");
//		startingPoint[0] += 0.001;
//		for(int i=0; i<startingPoint.length; i++){
//			startingPoint[i] = 1;
//			System.out.printf("%.3f ",startingPoint[i]);
//		}
//		System.out.println();
//		printGradientAt(logLikelihood, startingPoint);
		
		long startTime = System.currentTimeMillis();
		if (useSGD) {
			GradientDescentMinimizer sgdMinimizer = new GradientDescentMinimizer();
			sgdMinimizer.setLearningAdjustment(learningRate);
			sgdMinimizer.setIterations(iterations);
			sgdMinimizer.setEta0(eta0);
			if (batchSize == 0) batchSize = logLikelihood.staticTrainingData.size(); 
			sgdMinimizer.setBatchSize(batchSize);
			double scale = ((double)batchSize)/logLikelihood.staticTrainingData.size();
			lambdaReg = scale/Math.pow(regularizationParameter, 2);
			sgdMinimizer.setAlpha(lambdaReg);
			sgdMinimizer.setTimestep(timestep);
			sgdMinimizer.setGamma(gamma);
			
			sgdMinimizer.maxIterPrintAll = maxIterPrintAll;
			sgdMinimizer.interval = interval;
			
			weights = sgdMinimizer.minimize(logLikelihood, startingPoint);
		} else {
			Result result = null;
			try {
				System.out.println("Start maximizing Log-Likelihood...");
				// Actually minimize the negation of the actual log-likelihood
				// Which is equivalent to maximizing the actual log-likelihood
				result = minimize(logLikelihood, startingPoint);
			} catch (LBFGSBException e) {
				e.printStackTrace();
			}
			weights = result.point;
		}
		long endTime = System.currentTimeMillis();
		System.out.printf("Done minimizing in %.3fs\n", (endTime-startTime)/1000.0);
		
		System.out.println("Weights:");
		for(int i=0; i<weights.length; i++){
			System.out.printf("%.3f ", weights[i]);
		} 
		System.out.println("Done!");
	}

//	private void printGradientAt(LogLikelihood logLikelihood, double[] startingPoint) {
//		System.out.println("Gradient:");
//		FunctionValues values = logLikelihood.getValues(startingPoint);
//		double[] gradient = negate(values.gradient);
//		for(int i=0; i<gradient.length; i++){
//			System.out.printf("%.3f ", gradient[i]);
//		}
//		System.out.println();
//		System.out.printf("Value: %f\n", -values.functionValue);
//	}
	
	@Override
	public List<Instance> predict(List<Instance> testData) {
		List<Instance> results = new ArrayList<Instance>();
		buildFeatures(testData, false);
		for(Instance instance: testData){
			int n = instance.words.size()+2;
			int[][] parentIdx = new int[n][tags.size()];
			double[][] lattice = new double[n][tags.size()];
			lattice[0][tags.get(START)] = 1;
			fillValues(instance, lattice, true, weights, Common::max, parentIdx);
			
			int wordCount = n-1;
			List<TaggedWord> result = new ArrayList<TaggedWord>();
			int curIdx = parentIdx[wordCount][tags.get(END)];
			for(wordCount = n-2; wordCount >= 1; wordCount--){
				result.add(0, new TaggedWord(instance.words.get(wordCount-1).features(), reverseTags[curIdx]));
				curIdx = parentIdx[wordCount][curIdx];
			}
			results.add(new Instance(result));
		}
		return results;
	}
}